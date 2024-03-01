import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import logging
from eleet_pretrain.utils import compute_span_distances

logger = logging.getLogger(__name__)


class MMValue(list):

    def __str__(self):
        try:
            return str(self[0][0])
        except IndexError:
            return ""
    
    def iter_values(self):
        return iter(x[0] for x in self)

    def iter_embeddings(self):
        return iter(x[1] for x in self)

    def deduplicate(self, model, return_assignment=False, linkage="average"):
        values = list(self.iter_values())
        if len(values) < 2:
            if return_assignment:
                return self, np.array([0] * len(values))
            else:
                return self

        new_values = list()
        new_embeddings = list()
        with torch.no_grad():
            embeddings = torch.vstack(tuple(self.iter_embeddings()))

            c = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage=linkage,
                                        distance_threshold=0.0)
            distances = compute_span_distances(embeddings=embeddings, 
                                        duplicate_detect_threshold=model.duplicate_detect_threshold,
                                        duplicate_detect_layer=model.duplicate_detect_layer)
            assignment = c.fit_predict(distances.cpu().numpy())

            for a in sorted(set(assignment)):
                options = [values[v] for v in np.where(assignment == a)[0]]
                options_iter = iter(sorted(options, key=lambda x: (x.count("#"), -len(x))))
                selected = next(options_iter)
                not_selected = list(options_iter)
                if not_selected:
                    logger.debug(f" {selected} predicted to be same as {not_selected}.")
                new_values.append(selected)
                new_embeddings.append(embeddings[assignment == a].mean(0))
        self.clear()
        self.extend(zip(new_values, new_embeddings))
        if return_assignment:
            return self, assignment
        return self

    def __add__(self, other):
        return MMValue(super().__add__(other))
