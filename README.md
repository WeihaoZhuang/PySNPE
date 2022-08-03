# PySNPE
A lightweight python framework of SNPE (Snapdragon Neural Processing Engine https://developer.qualcomm.com/sites/default/files/docs/snpe/index.html)

### Highlight

We can use python scripts to convert the onnx model into a dlc model and perform model inference.

```
#Onnx model to DLC model
converter = OnnxConverter(model_path, host_snpe)
model = converter.onnx_to_dlc()
model.upload_model(adbkey_path=adbkey_path)

#Prepare input
inp = cv2.imread("./cat.jpeg").astype("float32")/255.
inp = np.ascontiguousarray(cv2.resize(inp[:,:,::-1],(224,224))[None])
inputs = ['input.1',inp]
inp_array = SnpeArray(inputs, mobile=True)

#Model inference
df,output = model(inp_array, 'gpu', profile=True)
```


### Requirements
SNPE:\
Please flow the web page to install SNPE SDK https://developer.qualcomm.com/sites/default/files/docs/snpe/usergroup0.html
    
Python ADB tool :\
https://github.com/JeffLIrion/adb_shell

### Tutorial

Please check [demo.ipynb](https://github.com/WeihaoZhuang/PySNPE/blob/master/demo.ipynb).

### TODO List
 - [ ] DSP runtime
    - The author's device is missing /system/vendor/lib/rfsa/adsp and /dsp folder. Could not run the model on DSP.  Although https://developer.qualcomm.com/forum/qdn-forums/software/qualcomm-neural-processing-sdk/68002 mentions that the missing files can be installed using HEXAGON SDK, it seems to require root access. Rooting the author's device is diffcult for some reasons.
