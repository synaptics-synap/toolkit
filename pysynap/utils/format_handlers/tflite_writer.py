# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2019 Synaptics Incorporated.

from pysynap.graph_segmenter import GlobalTensorId
from pysynap.external.tflite.reflection.BaseType import BaseType
from pysynap.external.tflite.reflection.Enum import Enum
from pysynap.external.tflite.reflection.Field import Field
from pysynap.external.tflite.reflection.Schema import Schema

from pysynap.external.tflite.tflite.Model import Model

import flatbuffers
from pathlib import Path
import importlib

from pysynap.network import OperationId, TensorId

import logging

logger = logging.getLogger("synap.tflite_writer")


class TfliteWriter:
    """
    Writes out a model object in the tflite format
    """

    def __init__(self):
        """
        Initializes the writer by loading the schema and creating the flatbuffers builder
        """
        with (Path(__file__).parent.parent.parent / 'external' / 'tflite' / 'schema.bfbs').open('rb') as fp:
            self.schema = Schema.GetRootAsSchema(bytearray(fp.read()), 0)
        self.builder = flatbuffers.Builder(1024)

        self.type_to_prepend_method = {
            BaseType.ULong: lambda x: self.builder.PrependUint64(x),
            BaseType.Int: lambda x: self.builder.PrependInt32(x),
            BaseType.UInt: lambda x: self.builder.PrependUint32(x),
            BaseType.Long: lambda x: self.builder.PrependInt64(x),
            BaseType.Bool: lambda x: self.builder.PrependBool(x),
            BaseType.Float: lambda x: self.builder.PrependFloat32(x),
            BaseType.Double: lambda x: self.builder.PrependFloat64(x),
        }

    def _prepend_value(self, base_type: BaseType, value):
        """
        Writes out the simple value of a field in the builder.
        """

        if base_type == BaseType.ULong:
            self.builder.PrependUint64(value)
        elif base_type == BaseType.Int:
            self.builder.PrependInt32(value)
        elif base_type == BaseType.UInt:
            self.builder.PrependUint32(value)
        elif base_type == BaseType.Long:
            self.builder.PrependInt64(value)
        elif base_type == BaseType.Bool:
            self.builder.PrependBool(value)
        elif base_type == BaseType.Float:
            self.builder.PrependFloat32(value)
        elif base_type == BaseType.Double:
            self.builder.PrependFloat64(value)
        else:
            raise Exception(f"Unsupported base type {base_type}")

    @staticmethod
    def _convert_field_name(field_name):
        """
        Fields in the schema are written in snake case, this function converts them to camel case
        """
        return "".join([x.capitalize() for x in field_name.decode().split("_")])

    def get_vector_length(self, obj, obj_mod, field: Field, path: list[int | str], field_name: str):
        """
        Return the vector length for the given target vector
        """

        return getattr(obj, field_name + "Length")()

    def get_vector_value(self, obj, obj_mod, field: Field, path: list[int | str], field_name: str, idx: int):
        """
        Returns the value of the specified vector item
        """
        return getattr(obj, field_name)(idx)

    def _write_field_value_vector(self, obj, obj_mod, field: Field, path: list[int | str]):
        """
        Writes out a vector paying attention to the case in which the vector is not present.
        """

        field_name = self._convert_field_name(field.Name())

        current_path = path + [field_name]

        # check if the vector field is actually set, we need to check this in the _tab because the API doesn't
        # allow us to distinguish between a length 0 vector and no vector at all
        o = flatbuffers.number_types.UOffsetTFlags.py_type(obj._tab.Offset(field.Offset()))
        if o == 0:
            return 0

        vector_length = self.get_vector_length(obj, obj_mod, field, current_path, field_name)

        entry_type = field.Type().Element()
        start_vector = getattr(obj_mod, type(obj).__name__ + "Start" + field_name + "Vector")

        if entry_type == BaseType.String:

            offsets = []
            for idx in range(vector_length):
                value = self.get_vector_value(obj, obj_mod, field, current_path, field_name, idx)
                offsets.append(self.builder.CreateString(value))

            start_vector(self.builder, vector_length)

            for offset in offsets[::-1]:
                self.builder.PrependUOffsetTRelative(offset)

            return self.builder.EndVector()

        elif entry_type == BaseType.Obj:

            offsets = []
            for idx in range(vector_length):
                value = self.get_vector_value(obj, obj_mod, field, current_path, field_name, idx)
                offsets.append(self._write_object(value, current_path + [idx]))

            start_vector(self.builder, vector_length)

            for offset in offsets[::-1]:
                self.builder.PrependUOffsetTRelative(offset)

            return self.builder.EndVector()

        elif entry_type == BaseType.UByte or entry_type == BaseType.Byte:

            start_vector(self.builder, vector_length)
            get_value_as_numpy = getattr(obj, field_name + "AsNumpy")

            values = get_value_as_numpy()

            if isinstance(values, int):
                return self.builder.EndVector()

            self.builder.head = self.builder.head - vector_length
            self.builder.Bytes[self.builder.head:(self.builder.head + vector_length)] = values.tobytes()

            return self.builder.EndVector()

        else:

            start_vector(self.builder, vector_length)

            for idx in range(vector_length)[::-1]:
                value = self.get_vector_value(obj, obj_mod, field, current_path, field_name, idx)
                self._prepend_value(entry_type, value)

            return self.builder.EndVector()

    def _find_union_type(self, index: int) -> Enum:
        """
        Find the union type from the index provided in the type, this is then used to find the effective type
        of a field X using the corresponding XType field
        """
        return self.schema.Enums(index)

    def get_field_value(self, obj, path: list[int | str], field_name: str):
        """
        Returns the value for a given field
        """
        return getattr(obj, field_name)()

    def _write_field_value(self, obj, obj_mod, field: Field, path: list[int | str]):
        """
        Write out the value of a field in the builder.
        """

        field_name = self._convert_field_name(field.Name())

        base_type = field.Type().BaseType()

        if base_type == BaseType.Vector:
            return self._write_field_value_vector(obj, obj_mod, field, path)

        current_path = path + [field_name]

        value = self.get_field_value(obj, current_path, field_name)

        # the field is not present, we return None
        if value is None:
            return None

        if base_type == BaseType.String:
            return self.builder.CreateString(value)
        elif base_type in [BaseType.ULong, BaseType.Long, BaseType.Int, BaseType.UInt, BaseType.Byte, BaseType.UByte,
                           BaseType.Float, BaseType.Double, BaseType.Bool, BaseType.UType]:
            return value
        elif base_type == BaseType.Union:
            # find the index of the current type in the Union
            value_type_idx = getattr(obj, field_name + "Type")()

            # find the type of the value based on the union index
            union_type = self._find_union_type(field.Type().Index())
            value_type_name = union_type.Values(value_type_idx).Name().decode()
            value_mod = self._load_object_mod(value_type_name)
            value_class = getattr(value_mod, value_type_name)

            # create an object for the value so that we can serialize it with the normal object serialization function
            value_obj = value_class()
            value_obj.Init(value.Bytes, value.Pos)

            offset = self._write_object(value_obj, current_path)

            return offset
        elif base_type == BaseType.Obj:
            return self._write_object(value, current_path)
        else:
            raise Exception(f"Unsupported type {base_type} in field {field_name} of object {current_path}")

    @staticmethod
    def _load_object_mod(obj_name: str):
        """
        Loads the python API of the given object by the name of its class
        """
        return importlib.import_module("pysynap.external.tflite.tflite." + obj_name)

    def _write_object(self, obj, path: list[int | str]):
        """
        Write out an object. This consists of writing out all the fields values first and the write out each field.
        """
        obj_schema = self._get_object_schema(obj)
        object_mod = self._load_object_mod(type(obj).__name__)

        values_or_offsets = []

        for field_idx in range(obj_schema.FieldsLength()):
            field = obj_schema.Fields(field_idx)

            # skip deprecated fields as the Python API doesn't contain them
            if field.Deprecated():
                values_or_offsets.append(None)
            else:
                value = self._write_field_value(obj, object_mod, field, path)
                values_or_offsets.append(value)

        getattr(object_mod, type(obj).__name__ + "Start")(self.builder)

        for field_idx in range(obj_schema.FieldsLength()):

            # skip fields that we detected are not present in the object being serialized
            if values_or_offsets[field_idx] is None:
                continue

            field = obj_schema.Fields(field_idx)

            field_name = self._convert_field_name(field.Name())

            add_field = getattr(object_mod, type(obj).__name__ + "Add" + field_name)

            add_field(self.builder, values_or_offsets[field_idx])

        return getattr(object_mod, type(obj).__name__ + "End")(self.builder)

    def _get_object_schema(self, obj):
        """
        Returns the schema for the specified object by looking up the schema for the name of the object class
        """
        name = ("tflite." + type(obj).__name__).encode()

        for idx in range(self.schema.ObjectsLength()):
            if self.schema.Objects(idx).Name() == name:
                return self.schema.Objects(idx)
        else:
            raise Exception(f"Object {name} not found in schema")

    def write_model(self, model: Model) -> bytearray:
        """
        Writes out the specified model into a bytearray
        """
        offset = self._write_object(model, [])
        self.builder.Finish(offset, self.schema.FileIdent())
        return self.builder.Output()


class FilteredTfliteWriter(TfliteWriter):
    """
    TFliteWriter that writes out only a selected part of a model
    """

    def __init__(self, buffers: list[int], tensors: list[int],
                 operations: list[int], inputs: list[int], outputs: list[TensorId]):
        """
        Initializes the writer by loading the schema and creating the flatbuffers builder
        """
        super().__init__()

        self.buffers_new_to_old = buffers
        self.tensors_new_to_old = tensors
        self.operations_new_to_old = operations
        self.inputs_new_to_old = inputs
        self.outputs_new_to_old = outputs

        self.buffers_old_to_new = {x: i for i, x in enumerate(self.buffers_new_to_old)}
        self.tensors_old_to_new = {x: i for i, x in enumerate(self.tensors_new_to_old)}
        self.operations_old_to_new = {x: i for i, x in enumerate(self.operations_new_to_old)}
        self.inputs_old_to_new = {x: i for i, x in enumerate(self.inputs_new_to_old)}
        self.outputs_old_to_new = {x: i for i, x in enumerate(self.outputs_new_to_old)}

        # the tensor -1 (unused optional input) is always mapped to -1
        self.tensors_old_to_new[-1] = -1

    def get_vector_length(self, obj, obj_mod, field: Field, path: list[int | str], field_name: str):
        """
        Change the vector length of the specified field to the length of the filtered list
        """
        match path:
            case ["Subgraphs", 0, "Inputs"]:
                return len(self.inputs_new_to_old)
            case ["Subgraphs", 0, "Outputs"]:
                return len(self.outputs_new_to_old)
            case ["Subgraphs", 0, "Tensors"]:
                return len(self.tensors_new_to_old)
            case ["Subgraphs", 0, "Operators"]:
                return len(self.operations_new_to_old)
            case ["Buffers"]:
                return len(self.buffers_new_to_old)
            case _:
                return super().get_vector_length(obj, obj_mod, field, path, field_name)

    def get_vector_value(self, obj, obj_mod, field: Field, path: list[int | str], field_name: str, idx: int):
        """
        Return the value of the specified vector item, remapping input, outputs, tensors, operations and buffers
        based on what was requested to keep in the resulting graph
        """

        match path:
            case ["Subgraphs", 0, "Inputs"]:
                return self.tensors_old_to_new[self.inputs_new_to_old[idx]]
            case ["Subgraphs", 0, "Outputs"]:
                return self.tensors_old_to_new[self.outputs_new_to_old[idx]]
            case ["Subgraphs", 0, "Tensors"]:
                return super().get_vector_value(obj, obj_mod, field, path, field_name, self.tensors_new_to_old[idx])
            case ["Subgraphs", 0, "Operators"]:
                return super().get_vector_value(obj, obj_mod, field, path, field_name, self.operations_new_to_old[idx])
            case ["Buffers"]:
                return super().get_vector_value(obj, obj_mod, field, path, field_name, self.buffers_new_to_old[idx])
            case ["Subgraphs", 0, "Operators", _, "Inputs"]:
                return self.tensors_old_to_new[super().get_vector_value(obj, obj_mod, field, path, field_name, idx)]
            case ["Subgraphs", 0, "Operators", _, "Outputs"]:
                return self.tensors_old_to_new[super().get_vector_value(obj, obj_mod, field, path, field_name, idx)]
            case ["Subgraphs", 0, "Operators", _, "Intermediates"]:
                return self.tensors_old_to_new[super().get_vector_value(obj, obj_mod, field, path, field_name, idx)]
            case _:
                return super().get_vector_value(obj, obj_mod, field, path, field_name, idx)

    def get_field_value(self, obj, path: list[int | str], field_name: str):
        """
        Return the field value, for buffer references make sure we remap the ID
        """
        match path:
            case ["Subgraphs", 0, "Tensors", _, "Buffer"]:
                return self.buffers_old_to_new[super().get_field_value(obj, path, field_name)]
            case _:
                return super().get_field_value(obj, path, field_name)


def _find_required_tensors(model: Model, operations: list[int], inputs: list[int] | None = None,
                           outputs: list[int] = None) -> list[int]:
    """
    Finds the list of tensors required to compute the specified list of operations
    """
    tensors = set(inputs) | set(outputs)

    for operation in operations:
        op = model.Subgraphs(0).Operators(int(operation))

        inputs = op.InputsAsNumpy()

        if not isinstance(inputs, int):
            for input_idx in inputs.tolist():
                if input_idx != -1:
                    tensors.add(input_idx)

        outputs = op.OutputsAsNumpy()

        if not isinstance(outputs, int):
            for output_idx in outputs.tolist():
                tensors.add(output_idx)

        intermediate = op.IntermediatesAsNumpy()

        if not isinstance(intermediate, int):
            for intermediate_idx in intermediate.tolist():
                tensors.add(intermediate_idx)

    # sort the tensors to try to keep the original order
    return sorted(tensors)


def _find_required_buffers(model: Model, tensors: list[int]) -> list[int]:
    """
    Finds the list of buffers required by the listed tensors
    """

    # sort the output to try to keep the original order
    return sorted(list({model.Subgraphs(0).Tensors(tensor).Buffer() for tensor in tensors} | {0}))


def write_model(model: Model, operations: list[OperationId] | None = None, inputs: list[TensorId] | None = None,
                outputs: list[TensorId] = None) -> tuple[bytearray, dict[TensorId, GlobalTensorId]]:
    """
    Writes out the specified model while keeping only the specified operations and setting the input/outputs
    to the specified tensors, returns the model as bytearray and a mapping from new to original tensor indices.
    """

    if operations is None:
        operations = [x for x in range(model.Subgraphs(0).OperatorsLength())]
    else:
        operations = [int(x) for x in operations]

    if inputs is None:
        inputs = [model.Subgraphs(0).Inputs(x) for x in range(model.Subgraphs(0).InputsLength())]
    else:
        inputs = [int(x) for x in inputs]

    if outputs is None:
        outputs = [model.Subgraphs(0).Outputs(x) for x in range(model.Subgraphs(0).OutputsLength())]
    else:
        outputs = [int(x) for x in outputs]

    required_tensors = _find_required_tensors(model, operations, inputs, outputs)
    required_buffers = _find_required_buffers(model, required_tensors)

    logger.debug("Requested operations: %s", operations)
    logger.debug("Requested inputs: %s", inputs)
    logger.debug("Requested outputs: %s", outputs)
    logger.debug("Required tensors: %s", required_tensors)
    logger.debug("Required buffers: %s", required_buffers)

    writer = FilteredTfliteWriter(required_buffers, required_tensors, operations, inputs, outputs)

    data = writer.write_model(model)

    new_to_old_idx = {str(y): str(x) for x, y in enumerate(required_tensors)}

    return data, new_to_old_idx
