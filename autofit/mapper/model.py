import copy
from typing import Optional, Union, Tuple

from autofit.mapper.model_object import ModelObject
from autofit.tools.pipeline import ResultsCollection
from autofit.tools.promise import AbstractPromise

RECURSION_LIMIT = 100


class AbstractModel(ModelObject):
    def __add__(self, other):
        instance = self.__class__()

        def add_items(item_dict):
            for key, value in item_dict.items():
                if isinstance(value, list) and hasattr(instance, key):
                    setattr(instance, key, getattr(instance, key) + value)
                else:
                    setattr(instance, key, value)

        add_items(self.__dict__)
        add_items(other.__dict__)
        return instance

    def copy(self):
        return copy.deepcopy(self)

    def populate(self, collection):
        return populate(self, collection)

    def object_for_path(self, path: (str,)) -> object:
        """
        Get the object at a given path.

        Parameters
        ----------
        path
            A tuple describing the path to an object in the model tree

        Returns
        -------
        object
            The object
        """
        instance = self
        for name in path:
            instance = getattr(instance, name)
        return instance

    def path_instance_tuples_for_class(self, cls: type, ignore_class=None):
        """
        Tuples containing the path tuple and instance for every instance of the class
        in the model tree.

        Parameters
        ----------
        ignore_class
            Children of instances of this class are ignored
        cls
            The type to find instances of

        Returns
        -------
        path_instance_tuples: [((str,), object)]
            Tuples containing the path to and instance of objects of the given type.
        """
        return path_instances_of_class(self, cls, ignore_class=ignore_class)

    def direct_tuples_with_type(self, class_type):
        return list(
            filter(
                lambda t: t[0] != "id" and isinstance(t[1], class_type),
                self.__dict__.items(),
            )
        )

    def attribute_tuples_with_type(self, class_type, ignore_class=None):
        return [
            (t[0][-1], t[1])
            for t in self.path_instance_tuples_for_class(
                class_type, ignore_class=ignore_class
            )
        ]


def populate(
        obj,
        collection: ResultsCollection,
        recursion_depth=0
):
    """
    Replace promises with instances and constants. Promises are placeholders expressing that a given attribute should
    be replaced with an actual value once the phase that generates that value is complete.

    Parameters
    ----------
    obj
        The object to be populated
    collection
        A collection of Results from previous phases
    recursion_depth
        Current depth of recursion used to prevent infinite recursion

    Returns
    -------
    obj
        The same object with all promises populated, or if the object was a promise the replacement for that promise
    """
    if recursion_depth > RECURSION_LIMIT:
        raise RecursionError(
            f"Recursion limit {RECURSION_LIMIT} exceeded populating {obj}"
        )
    if isinstance(obj, list):
        return [
            populate(
                item,
                collection,
                recursion_depth=recursion_depth + 1
            ) for item in obj
        ]
    if isinstance(obj, dict):
        return {
            key: populate(
                value,
                collection,
                recursion_depth=recursion_depth + 1
            ) for key, value
            in obj.items()
        }
    if isinstance(obj, AbstractPromise):
        return obj.populate(collection)
    try:
        new = copy.deepcopy(obj)
        for key, value in obj.__dict__.items():
            setattr(new, key, populate(
                value,
                collection,
                recursion_depth=recursion_depth + 1
            ))
        return new
    except (AttributeError, TypeError, RecursionError):
        return obj


def path_instances_of_class(
        obj,
        cls: type,
        ignore_class: Optional[
            Union[type, Tuple[type]]
        ] = None,
        recursion_depth: int = 0
):
    """
    Recursively search the object for instances of a given class

    Parameters
    ----------
    obj
        The object to recursively search
    cls
        The type to search for
    ignore_class
        A type or
    recursion_depth
        Keeps track of the number of recursions made to ensure that cycles are broken

    Returns
    -------
    instance of type
    """
    if recursion_depth > RECURSION_LIMIT:
        raise RecursionError(
            f"Recursion searching for instances of {cls} in {obj} exceeded {RECURSION_LIMIT}"
        )
    if ignore_class is not None and isinstance(obj, ignore_class):
        return []
    if isinstance(obj, cls):
        return [(tuple(), obj)]
    results = []
    try:
        from autofit.mapper.prior_model.annotation import AnnotationPriorModel
        for key, value in obj.__dict__.items():
            try:
                for item in path_instances_of_class(
                        value,
                        cls,
                        ignore_class=ignore_class,
                        recursion_depth=recursion_depth + 1
                ):
                    if isinstance(value, AnnotationPriorModel):
                        path = (key,)
                    else:
                        path = (key, *item[0])
                    results.append((path, item[1]))
            except RecursionError:
                pass
        return results
    except AttributeError:
        return []


class ModelInstance(AbstractModel):
    """
    An object to hold model instances produced by providing arguments to a model mapper.

    @DynamicAttrs
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getitem__(self, item):
        return self.items[item]

    @property
    def items(self):
        return list(self.dict.values())

    @property
    def dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("id", "component_number", "item_number")
        }

    def __len__(self):
        return len(self.items)

    def as_variable(self, variable_classes=tuple()):
        from autofit.mapper.prior_model.abstract import AbstractPriorModel

        return AbstractPriorModel.from_instance(self, variable_classes)
