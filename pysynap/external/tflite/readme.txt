To download the schema::

    wget https://raw.githubusercontent.com/tensorflow/tensorflow/v2.14.0/tensorflow/lite/schema/schema.fbs

To generate the tflite directory containing the tflite schema API::

    flatc --python schema.fbs

To download the reflection schema::

   wget https://raw.githubusercontent.com/google/flatbuffers/v1.11.0/reflection/reflection.fbs

To create the reflection directory containing the reflection schema API::

   flatc --python reflection.fbs

To create the reflection description of the tflite schema::

   flatc -b --schema schema.fbs
