from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(input_saved_model_dir="./ssd_mobilenet_v2_2")
converter.convert()
converter.save("./ssd_mobilenet_v2_trt")