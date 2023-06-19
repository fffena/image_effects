import onnxruntime as ort
 
print(ort.get_available_providers())
 
provider = ['CUDAExecutionProvider','CPUExecutionProvider']
ort_sess = ort.InferenceSession(r"C:\Users\Taiki\Downloads\mobilenetv2-7.onnx", providers=provider)
print(ort_sess.get_providers())