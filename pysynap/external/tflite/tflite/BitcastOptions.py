# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class BitcastOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBitcastOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BitcastOptions()
        x.Init(buf, n + offset)
        return x

    # BitcastOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def BitcastOptionsStart(builder): builder.StartObject(0)
def BitcastOptionsEnd(builder): return builder.EndObject()
