from collections import defaultdict
import inspect
import torch
import numpy as np
import logging
from eleet_pretrain.datasets.dataset import EleetInferenceDataset
from eleet_pretrain.model.table_decoding_dependencies import DependencyTree
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

from eleet_pretrain.utils import DebugUnderlining, debug_transform, visualize_single, compute_span_distances
from copy import deepcopy
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class EleetInferenceDataLoader(DataLoader):
    """Data Loader that takes column dependencies into account and implements the algorithm for complex db ops."""
    def __init__(self,
                 eval_dataset,
                 is_complex_operation,
                 batch_size,
                 collate_fn,
                 drop_last,
                 num_workers,
                 pin_memory,
                 model):

        assert isinstance(eval_dataset, EleetInferenceDataset)
        self.is_complex_operation = is_complex_operation
        self._dataset = None
        self.orig_dataset = eval_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.debug_fraction = model.debug_fraction
        self.tokenizer = model.tokenizer
        self.union_mode = model.union_mode
        self.config = model.config
        self.dependency_trees = dict()
        self.tree_iterators = dict()
        self.current_batch_descs = []
        self.current_data = dict()
        self.current_loader = None
        self.current_stage = 0
        self.rng = np.random.default_rng(42)
        self.fast_mode = model.fast_mode
        self.duplicate_detect_layer = deepcopy(model.duplicate_detect_layer).to("cpu")
        self.duplicate_detect_threshold = model.duplicate_detect_threshold.detach().clone().to("cpu")
        self._load()

    def _load(self):
        """Sets up the data loader, especially the dependencies between columns."""
        loader = DataLoader(self.orig_dataset, batch_size=1, shuffle=None,
                            num_workers=self.num_workers, collate_fn=self.collate_fn)
        collected_model_in, collected_batch_descs = list(), dict()
        for x in loader:
            table_id = self._get_table_id(x)
            tree = self.dependency_trees[table_id] = DependencyTree.construct_from_query_coords(
                input_ids=x["input_ids"], query_coords=x["query_coords"],
                config=self.config, is_eval=True, tokenizer=self.tokenizer,
                duplicate_detect_layer=self.duplicate_detect_layer,
                duplicate_detect_threshold=self.duplicate_detect_threshold,
                enable_learned_deduplication=self.config.enable_learned_deduplication,
                is_complex_operation=self.is_complex_operation
            )
            self.tree_iterators[table_id] = iter(tree.iter_batches())
            self._collect_data(collected_model_in, collected_batch_descs, x, table_id, tree)
        self._set_current_data(collected_model_in, collected_batch_descs)

    def _get_table_id(self, x):
        """Transforms the data from the dataset into the format required by the model."""
        table_id = tuple(x["table_id"][0].tolist())
        return table_id

    def _set_current_data(self, collected_model_in, collected_batch_descs):
        """Sets the data for the current iteration of the complex db op algorithm."""
        self.current_data = dict()
        self.current_batch_descs = dict()
        if len(collected_model_in) > 0:
            for key in collected_model_in[0]:
                value = torch.vstack([c[key] for c in collected_model_in])
                self.current_data[key] = value
        self.current_batch_descs = collected_batch_descs
        self._dataset = TempBatchedDataset(self.current_data)
        batch_args = dict(batch_size=self.batch_size, drop_last=self.drop_last, shuffle=None)
        if self.fast_mode and len(collected_model_in) > 0:
            batch_args = {"batch_sampler": FastModeBatchSampler(self.current_data["table_id"][:, 1], **batch_args)}
        self.current_loader = DataLoader(
            self._dataset, **batch_args,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def _collect_data(self, collected_model_in, collected_batch_descs, x, table_id, tree):
        """Collects the data for the current iteration of the complex db op algorithm."""
        iterator = self.tree_iterators[table_id]
        collected_batch_descs[table_id] = list()
        for i, (batch_desc, query_mask) in enumerate(iterator):
            if batch_desc == tree.SIG_ITERATOR_DONE:
                break
            model_in, sd_in = tree.compute_model_inputs(
                    batch_desc=batch_desc,
                    query_mask=query_mask,
                    column_token_position_to_column_ids=x["column_token_position_to_column_ids"],
                    context_token_mask=x["context_token_mask"],
                    context_token_positions=x["context_token_positions"],
                    input_ids=x["input_ids"],
                    sequence_mask=x["sequence_mask"],
                    table_mask=x["table_mask"],
                    token_type_ids=x["token_type_ids"],
                    query_labels=x["query_labels"],
                    query_coords=x["query_coords"],
                    do_remove_dependent=self.current_stage == 0 and not self.union_mode
            )
            model_in.update(sd_in)
            model_in["idx"] = x["idx"]
            model_in["table_id"] = x["table_id"]
            model_in["sub_idx"] = torch.full_like(x["idx"], i)
            model_in["stage"] = torch.full_like(x["idx"], self.current_stage)
            collected_model_in.append(model_in)
            collected_batch_descs[table_id].append((x["idx"], i, batch_desc))

    def __iter__(self):
        return iter(self.current_loader)

    def advance_stage(self, idx, table_ids, sub_idx, sd_pred, embeddings):
        """Advances the current stage of the complex db op algorithm."""
        current_data_permuted = self.permute_current_data(idx, sub_idx)
        self.current_stage += 1
        with torch.no_grad():
            loader = DataLoader(self.orig_dataset, batch_size=1, shuffle=None,
                                num_workers=self.num_workers, collate_fn=self.collate_fn)
            collected_model_in, collected_batch_descs = list(), dict()

            for x in loader:
                current_table_id = self._get_table_id(x)
                tree = self.dependency_trees[current_table_id]
                table_id_mask = (table_ids == torch.tensor(current_table_id)).all(1)
                for current_idx, current_sub_idx, batch_desc in self.current_batch_descs[current_table_id]:
                    # mask to select the queries for the current iteration
                    mask = (current_idx.T == idx).any(1)
                    mask = mask & table_id_mask
                    mask = mask & (sub_idx == current_sub_idx).all(1)
                    self.stage_debug_print(current_data_permuted, sd_pred, mask)
                    tree.update(batch_desc,
                        current_data_permuted["input_ids"][mask], sd_pred[mask],
                        current_data_permuted["query_mask"][mask],
                        embeddings=embeddings[mask]
                    )

                self._collect_data(collected_model_in, collected_batch_descs, x, current_table_id, tree)
        self._set_current_data(collected_model_in, collected_batch_descs)

    def permute_current_data(self, idx, sub_idx):
        """Bring the data in the right order."""
        i1 = (idx * (sub_idx.max() + 1) + sub_idx).view(-1).sort().indices
        i1_rev = i1.sort().indices
        i2 = (self.current_data["idx"] * (self.current_data["sub_idx"].max() + 1)
              + self.current_data["sub_idx"]).view(-1).sort().indices
        current_data_permuted = {k: v[i2][i1_rev] for k, v in self.current_data.items()}
        assert (current_data_permuted["idx"] == idx).all()
        return current_data_permuted

    def is_done(self):
        """Returns true if the complex db op algorithm is done."""
        return len(self) == 0

    def get_final_results(self, return_embeddings=False):
        """Returns the final extractions and potentially their embeddings of the complex db op algorithm."""
        result = dict()
        if not return_embeddings:
            for table_id, tree in sorted(self.dependency_trees.items()):
                result_table = tree.construct_result_table(return_embeddings=False)
                result[table_id] = result_table
            return result

        result_embeddings = dict()
        for table_id, tree in sorted(self.dependency_trees.items()):
            result_table, result_table_embeddings = tree.construct_result_table(return_embeddings=True)
            result[table_id] = result_table
            result_embeddings[table_id] = result_table_embeddings
        return result, result_embeddings

    def final_results_eval(self, compute_metrics, other_labels=(), normalize=False):
        final_results = self.get_final_results(return_embeddings=normalize)
        replacements = None
        if normalize:
            final_results, replacements = self.normalize(*final_results)
        loader = DataLoader(self.orig_dataset, batch_size=1, shuffle=None,
                            num_workers=self.num_workers, collate_fn=self.collate_fn)
        preds, labels = list(), list()
        for x in loader:
            current_table_id = self._get_table_id(x)
            result_table = final_results[current_table_id]
            tree = self.dependency_trees[current_table_id]
            result_vector = tree.to_result_vector(result_table,
                                                  x["column_token_position_to_column_ids"], x["input_ids"])
            self._debug_print_result_tables(result_vector, prediction=True)
            self._debug_print_result_tables(x["result_table"])
            preds.append(result_vector)
            labels.append(x["result_table"])
        labels = torch.stack(labels)
        preds = torch.stack(preds)
        if other_labels is not None:
            labels = (labels, *other_labels)
        if "replacements" in inspect.getfullargspec(compute_metrics).args:
            metrics = compute_metrics((preds, labels), replacements=replacements)
        else:
            metrics = compute_metrics((preds, labels))
        return metrics

    def normalize(self, values, embeddings):
        """Normalizes the extracted values. Finds synonyms, clusters them together and picks the most common one."""
        replacements = defaultdict(set)
        collected_embeddings = self.normalize_collect_embeddings(values, embeddings)
        clusters = self.cluster_embeddings(collected_embeddings)
        for (col_set_id, col_nr, _), data in clusters.items():
            for normed_value, table_id, row_id, row_nr, value_nr in data:
                if isinstance(values[table_id, col_set_id][row_id][row_nr][col_nr], list):
                    old = values[table_id, col_set_id][row_id][row_nr][col_nr][value_nr]
                    values[table_id, col_set_id][row_id][row_nr][col_nr][value_nr] = normed_value
                else:
                    old = values[table_id, col_set_id][row_id][row_nr][col_nr] 
                    values[table_id, col_set_id][row_id][row_nr][col_nr] = normed_value

                decoded_old = self.tokenizer.decode(old)
                decoded_normed_value = self.tokenizer.decode(normed_value)
                replacements[decoded_normed_value].add(decoded_old)
                logger.debug(f"Normalization: Replaced {decoded_old} by {decoded_normed_value}")
        return values, replacements

    def cluster_embeddings(self, collected_embeddings):
        """Clusters the embeddings of the extracted values to find synonyms."""
        with torch.no_grad():
            result = dict()
            for col_id, data in collected_embeddings.items():
                value, embedding, table_id, row_id, row_nr, value_nr = zip(*data)
                if len(embedding) < 2:
                    continue
                embedding_matrix = torch.vstack(embedding)
                distances = compute_span_distances(embedding_matrix, self.duplicate_detect_layer, self.duplicate_detect_threshold)

                c = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average",
                                            distance_threshold=0.0)
                assignment = c.fit_predict(distances.cpu().numpy())

                groups = dict()
                for a, v in zip(assignment, value):
                    groups[a] = groups.get(a, list())
                    groups[a].append(v)
                groups = {k: max(v, key=len) for k, v in groups.items()}

                for x in zip(assignment, table_id, row_id, row_nr, value_nr):
                    a = x[0]
                    result[(*col_id, a)] = result.get((col_id, a), list())
                    result[(*col_id, a)].append((groups[a], *x[1:]))
            return result

    def normalize_collect_embeddings(self, values, embeddings):
        """Collects the embeddings of the extracted values."""
        result = dict()
        for (table_id, col_set_id), data in values.items():
            for row_id, rows in data.items():
                for i, row in enumerate(rows):
                    for col_nr, val in row.items():
                        result[col_set_id, col_nr] = result.get((col_set_id, col_nr), list())
                        emb = embeddings[table_id, col_set_id][row_id][i][col_nr]
                        if not isinstance(val, list):
                            val = [val]
                            emb = [emb]

                        for j, (v, e) in enumerate(zip(val, emb)):
                            result[col_set_id, col_nr].append((v, e, table_id, row_id, i, j))
        return result

    def get_labels(self):
        """Returns the labels of the dataset."""
        self.collate_fn.add_ground_truth = True
        loader = DataLoader(self.orig_dataset, batch_size=1, shuffle=None,
                            num_workers=self.num_workers, collate_fn=self.collate_fn)

        labels = list()
        table_ids = list()
        query_coords = list()
        for x in loader:
            self._debug_print_result_tables(x["result_table"])
            labels.append(x["result_table"])
            table_ids.append(x["table_id"][0])
            query_coords.append(x["query_coords"])
        labels = torch.stack(labels)
        table_ids = torch.stack(table_ids)
        self.collate_fn.add_ground_truth = False
        return labels, table_ids, query_coords

    def __len__(self):
        """Compute the number of elements."""
        return len(self.current_loader)

    def _debug_print_result_tables(self, result_table, prediction=False):
        """Prints the result tables."""
        if logger.root.level > logging.DEBUG:
            return
        result = ""
        for j, row_table in enumerate(result_table):
            if (row_table > -1).any():
                prev = 0
                for boundary in torch.where(row_table == -1)[0]:
                    if boundary == prev:
                        prev = boundary + 1
                        continue
                    result += f"id | {j} [SEP] "
                    result += self.tokenizer.decode(row_table[prev: boundary])
                    prev = boundary + 1
                    result += "\n"
        result += "\n"
        logger.debug(f"{'PREDICTED' if prediction else 'GROUND TRUTH'} FULL RESULTS:")
        logger.debug("\n" + result)

    def stage_debug_print(self, current_data, sd_pred, mask):
        """Prints the debug information after each iteration of the complex db ops algorithm."""
        if logging.root.level <= logging.DEBUG and self.rng.random() < self.debug_fraction:
            current_data = {k: v[mask] for k, v in current_data.items()}
            sd_pred = sd_pred[mask]

            self._stage_debug_print(input_ids=current_data["input_ids"],
                                    token_type_ids=current_data["token_type_ids"],
                                    sequence_mask=current_data["sequence_mask"],
                                    query_labels=current_data["query_labels"],
                                    query_coords=current_data["query_coords"],
                                    sd_pred=sd_pred)

    def _stage_debug_print(self, input_ids, token_type_ids, sequence_mask, query_labels,
                           query_coords, sd_pred):
        """Prints the debug information after each iteration of the complex db ops algorithm."""
        for i in range(input_ids.size(0)):
            answer_start, answer_end, answer_col_ids = debug_transform(input_ids[i], query_labels[i], query_coords[i])
            pred_start, pred_end, pred_col_ids = debug_transform(input_ids[i], sd_pred[i], query_coords[i])

            visualize_single(
                self.tokenizer,
                input_ids=input_ids[i],
                token_type_ids=token_type_ids[i],
                sequence_mask=sequence_mask[i],
                is_training=False,
                print_func=logger.debug,
                underlinings=(
                    DebugUnderlining("Masked Cells", "M", answer_start, answer_end, answer_col_ids),
                    DebugUnderlining("Prediction", "P", pred_start, pred_end, pred_col_ids),
                )
            )


class FastModeBatchSampler(BatchSampler):
    """
    A batch sampler that allows to first perform the first iteration of the complex db ops algorithm on all
    database, and then advance all datapoints at once.
    """
    def __init__(self, col_ids, batch_size, drop_last=False, shuffle=None):
        if shuffle or drop_last:
            raise NotImplementedError
        self.col_ids = col_ids
        self.batch_samplers = []
        for v in col_ids.unique():
            sampler = SequentialSampler(col_ids[col_ids == v])
            self.batch_samplers.append((v, BatchSampler(sampler, batch_size, drop_last=False)))

    def __iter__(self):
        for col_id, bs in self.batch_samplers:
            sub = torch.where(self.col_ids == col_id)[0]
            for i in bs:
                yield sub[i].tolist()
        

    def __len__(self):
        return sum(len(bs) for _, bs in self.batch_samplers)


class TempBatchedDataset(Dataset):
    def __init__(self, current_data):
        self.current_data = current_data

    def __getitem__(self, idx):
        """Get an element from the dataset."""
        result = {
            key: val[idx] for key, val in self.current_data.items()
        }
        return result

    def __len__(self):
        """Compute the number of elements."""
        return 0 if "input_ids" not in self.current_data else len(self.current_data["input_ids"])
