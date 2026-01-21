Experiment modules should implement the following functions:

    def make_concept_model(config: dict) -> ConceptModel
        """
        Create a concept model from the given configuration.
        """
        ...
    
    def get_config(**kwargs) -> dict
        """
        Return the default configuration for this experiment
        (optionally overriding with provided keyword arguments).
        """
        ...
