from eleet_pretrain.datasets.base_loader import BaseLoader
from eleet_pretrain.steps import Step
from eleet_pretrain.datasets import WikidataPreprocessStep
import logging

logger = logging.getLogger(__name__)


NUM_GRAPH_TRAVERSAL_STEPS = 20


class WikidataHierarchyTraverser(BaseLoader):
    
    def compute_transitive_hull(self):
        self.mongo_connect()
        self.wikidata_store.create_indexes()
        logger.info("Computing wikidata class hierarchy")
        self.wikidata_store.aggregate([
            {"$match": {"superclasses.0": {"$exists": True}}},
            {"$project": {"superclasses": 1, "wikidata_id": 1}},
            {"$out": "wikidata-class-hierarchy"}
        ])
        self.wikidata_class_hierarchy_store.create_indexes()
        self.wikidata_class_hierarchy_store.aggregate([
            {"$set": {"superclasses0": ["$wikidata_id"]}},
            *([
                {"$lookup": {
                    "from": "wikidata-class-hierarchy",
                    "localField": "superclasses",
                    "foreignField": "wikidata_id",
                    "as":"superclasses1",
                    "pipeline": [{"$group": {"_id": 1, "x": {"$push": "$superclasses"}}}]
                }},
                {"$set": {"superclasses1": {"$ifNull": [{"$first": "$superclasses1.x"}, []]}}},
                {"$set": {"superclasses1": {"$reduce": {"input": "$superclasses1",
                                                        "initialValue": [],
                                                        "in": {"$setUnion": ["$$this", "$$value"]}
                }}}},
                {"$project": {"superclasses": {"$setDifference": ["$superclasses1", {"$setUnion": ["$superclasses",
                                                                                                   "$superclasses0"]}]},
                              "wikidata_id": 1,
                              "superclasses0": {"$setUnion": ["$superclasses", "$superclasses0"]}}}
            ] * NUM_GRAPH_TRAVERSAL_STEPS),
            {"$project": {"wikidata_id": 1, "transitive_superclasses": {"$setUnion": ["$superclasses",
                                                                                      "$superclasses0"]}}},
            {"$out": "wikidata-class-hierarchy"}
        ])
        self.wikidata_class_hierarchy_store.create_indexes()


class WikidataClassHierarchyStep(Step):
    """Load data from wikidata dump."""

    depends_on = {WikidataPreprocessStep}

    def check_done(self, args, dataset, **kwargs):
        """Check whether the step has already been executed."""
        x = WikidataHierarchyTraverser(dataset_dir=args.dataset_dir,
                                       sample=args.small_sample,
                                       dataset=dataset)
        x.mongo_connect()
        return not x.wikidata_class_hierarchy_store.is_empty()

    def run(self, args, dataset):
        """Execute the step."""
        x = WikidataHierarchyTraverser(dataset_dir=args.dataset_dir,
                                       sample=args.small_sample,
                                       dataset=dataset)
        x.compute_transitive_hull()
