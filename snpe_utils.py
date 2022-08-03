import os
import numpy as np
import pandas as pd
import subprocess
import onnx
from adb_helper import ADB
from typing import List, AnyStr, Tuple
from pathlib import Path
from onnx import shape_inference
class SnpeArray():
    def __init__(self, inputs: Tuple[str, np.array], mobile_folder: str="/data/local/tmp/snpe", mobile: bool=True):
        """
        Custom numpy array warpping for SNPE.
        Args:
        inputs (Tuple[str, np.array]): 
        
        mobile (bool):
        
        """
        self.input_names = inputs[::2]
        self.input_array = inputs[1::2]
        self.inputs = inputs
        self.host_inputs_folder = "/tmp/snpe/inputs/"
        self.mobile_folder = mobile_folder
        self.mobile = mobile
        self.raw_names = []
        self.dump_array2file()
        self.inputs_for_snpe()
        

    def dump_array2file(self, )->None:
        for idx,(name, array) in enumerate(zip(self.input_names, self.input_array)):
            p = f"{self.host_inputs_folder}/{name}_{idx}.raw"
            array.tofile(p)
        
    def inputs_for_snpe(self,) -> None:
        batch_size = self.input_array[0].shape[0]
        f =  open(f"{self.host_inputs_folder}/inputs.txt", "w")
        for b in range(batch_size):
            input_path = ""
            for name in self.input_names:
                raw_name = f"{name}_{b}.raw"
                if self.mobile:
                    input_path += f"{name}:={self.mobile_folder}/inputs/{raw_name} "
                else:
                    input_path += f"{name}:={self.host_inputs_folder}/{raw_name} "
                self.raw_names.append(raw_name)

            f.write(f"{input_path}\n")
        f.close()
        
class SnpeModel():
    def __init__(self, dlc_path: str,
                       inp_info: dict,
                       out_info:dict):
        """
        Args: 
        dlc_path:
            Path of SNPE dlc model
        inp_info:
            Dict of onnx model input info
        out_info:
            Dict of onnx model input info
        """
        self.dlc_path = dlc_path
        self.inp_info = inp_info
        self.out_info = out_info
        self.RUN_OPTIONS = {"aip": "--use_aip --platform_options unsignedPD:ON",
                            "dsp": "--use_dsp --platform_options unsignedPD:ON",
                            "gpu": "--use_gpu --gpu_mode=float16",
                            "cpu": ""}
        
    def quantize(self,inputs: SnpeArray,
                 no_weight_quantization: bool=False,
                 output_dlc: str="model_qua.dlc",
                 enable_hta: bool=False,
                 hta_partitions: bool=False,
                 enable_htp: bool=False,
                 htp_socs: str=None,
                 buffer_data_type: bool=False,
                 use_enhanced_quantizer: bool=False,
                 use_adjusted_weights_quantizer: bool=False,
                 optimizations: str=None,
                 override_params: bool=False,
                 use_symmetric_quantize_weights: bool=False,
                 bias_bitwidth: str=None,
                 act_bitwidth: str=None,
                 weights_bitwidth: str=None,
                 bitwidth: str=None,
                 udo_package_path: str=None
                ):
        """
        SNPE dlc model quantization, please check https://developer.qualcomm.com/sites/default/files/docs/snpe/tools.html#tools_snpe-dlc-quantize
        
        """
        cmd = ["snpe-dlc-quantize",
              "--input_list", inputs.txt_file,
              "--input_dlc", self.dlc_path,
              "--output_dlc", output_dlc]
        
        if no_weight_quantization:
            cmd += ["--no_weight_quantization"]
        if enable_hta:
            cmd += ["--enable_hta"]
        if hta_partitions:
            cmd += ["--hta_partitions"]
        if htp_socs:
            cmd += ["--htp_socs", htp_socs]
        if buffer_data_type:
            cmd += ["--buffer_data_type"]
        if use_enhanced_quantizer:
            cmd += ["--use_enhanced_quantizer"]
        if use_adjusted_weights_quantizer:
            cmd += ["--use_adjusted_weights_quantizer"]
        if optimizations:
            cmd += ["--optimizations", self.optimizations]
        if override_params:
            cmd += ["--override_params"]
        if use_symmetric_quantize_weights:
            cmd += ["--use_symmetric_quantize_weights"]
            
        if bias_bitwidth:
            cmd += ["--bias_bitwidth", bias_bitwidth]
        
        if act_bitwidth:
            cmd += ["--act_bitwidth", act_bitwidth]
        
        if weights_bitwidth:
            cmd += ["--weights_bitwidth", weights_bitwidth]
            
        if bitwidth:
            cmd += ["--bitwidth", bitwidth]
                
        if udo_package_path:
            cmd += ["--udo_package_path", udo_package_path]
                
            
        subprocess.call(cmd , env=os.environ)
        self.dlc_path = output_dlc
    
    def upload_model(self, adbkey_path, mobile_folder="/data/local/tmp/snpe"):
        self.adbkey_path = adbkey_path
        self.mobile_folder = mobile_folder
        adb = ADB(self.adbkey_path)
        model_name = Path(self.dlc_path).stem
        cmd_ret = adb.push(self.dlc_path, f"{mobile_folder}/models/{model_name}.dlc")
    
    
    def upload_inputs(self, inputs: SnpeArray):
        adb = ADB(self.adbkey_path)
        host_folder = inputs.host_inputs_folder
        for raw_name in inputs.raw_names:
            host_p = f"{host_folder}/{raw_name}"
            mobile_p = f"{self.mobile_folder}/inputs/{raw_name}"
            adb.push(host_p, mobile_p)
            
        adb.push(f"{host_folder}/inputs.txt", f"{self.mobile_folder}/inputs/inputs.txt")

    def run_net(self, runtime):
        ld   = f"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{self.mobile_folder}/lib:/system/vendor/lib/"
        path = f"PATH=$PATH:{self.mobile_folder}/bin"
        adsp = f'ADSP_LIBRARY_PATH="{self.mobile_folder}/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"'
        cmd = [ld,
               path,
               adsp,
               "snpe-net-run",
               "--container", f"{self.mobile_folder}/models/{Path(self.dlc_path).stem}.dlc",
               "--input_list", f"{self.mobile_folder}/inputs/inputs.txt",
               "--output_dir", f"{self.mobile_folder}/outputs/",
               self.RUN_OPTIONS[runtime]]

        cmd = " ".join(cmd)
        print(cmd)
        adb = ADB(self.adbkey_path)
        adb.shell(cmd)
        

    def get_outputs(self):
        adb = ADB(self.adbkey_path)
        ret = {}
        for name,shape in self.out_info.items():
            mobile_file = f"{self.mobile_folder}/outputs/Result_0/{name}.raw"
            host_file = f"/tmp/snpe/outputs/{name}.raw" 
            adb.pull(mobile_file, host_file)
            if len(shape)==4:
                shape = [shape[0],shape[2],shape[3],shape[1]]
            ret[name]=np.fromfile(host_file, "float32").reshape(*shape)
        adb.shell(f"rm -r {self.mobile_folder}/outputs/Results")
        adb.shell(f"rm {self.mobile_folder}/outputs/SNPEDiag_0.log")
        return ret
    
    def get_profile_log(self,):
        adb = ADB(self.adbkey_path)
        mobile_log = f"{self.mobile_folder}/outputs/SNPEDiag_0.log"
        host_csv = f"/tmp/snpe/outputs/SNPEDiag.log"
        adb.pull(mobile_log, host_csv)
        ret = subprocess.call(['snpe-diagview', '--input_log', host_csv, '--output', host_csv.replace("log","csv")], env=os.environ)
        df = pd.read_csv(host_csv.replace("log","csv"))
        df.columns = ["timestamp", "message", "idx", "time", "runtime"]
        df = df[df.idx>=0] 
        return df
    
    def __call__(self, inputs, runtime, profile=False):
        adb = ADB(self.adbkey_path)
        self.upload_inputs(inputs)
        self.run_net(runtime)
        
        if profile:
            return self.get_profile_log(), self.get_outputs()
        return self.get_outputs()
    
    
class OnnxConverter():
    def __init__(self, model_path: str, 
                       host_snpe: str, 
                       # out_name: List[str],
                       output_path: str="model.dlc",
                       copyright_file: str=None,
                       model_version: str=None,
                       input_type: List[List[str]]=[],
                       input_dtype: List[List[str]]=[],
                       input_encoding: List[List[str]]=[],
                       input_layout: List[List[str]]=[],
                       
                       no_simplification: bool=False,
                       disable_batchnorm_folding: bool=False,
                       keep_disconnected_nodes: bool=False,
                       validation_target: list=[],
                       strict: bool= False,
                       debug: bool=False,
                       dry_run: str=None,
                       udo_config_paths: List[str]=[],
                       quantization_overrides: bool=False,
                       keep_quant_nodes: bool=False
                       ):
        """
        Please check the https://developer.qualcomm.com/sites/default/files/docs/snpe/tools.html#tools_snpe-onnx-to-dlc for details.
        
        Args:
        model_path (str):
            Path of ONNX model.
        host_snpe (str):
            Path of SNPE SDK.
        out_name (List[str]):
            list of onnx output names 
        output_path (str):
            Path fo output DLC model
        copyright_file (str):
            
        model_version (str):
            
        no_simplification (bool):
            
        disable_batchnorm_folding (bool):
            
        keep_disconnected_nodes (bool):
        
        input_type (list(list[str,str])):
            Eg: [["data1", "image],["data2", "opaque"]]
            
        input_dtype (list(list[str,str])):
            Eg: [["data1", "float32],["data2", "data"]]
        
        input_encoding (list(list[str,str])):
            Eg: [["data1", "rgba],["data2", "rgb"]]
        
        input_layout (list(list[str,str])):
            Eg: [["data1", "NCDHW],["data2", "NHWC"]]
        
        validation_target (str):
        strict (bool):
        
        debug (bool):
        dry_run (bool)
        udo_config_paths (list[str]):
        quantization_overrides (bool):
        keep_quant_nodes (bool):
        """

        self.model_path = model_path
        self.output_path = output_path
        self.copyright_file = copyright_file
        self.model_version = model_version
        self.input_type = input_type
        self.input_dtype = input_dtype
        self.input_encoding = input_encoding
        self.input_layout = input_layout
        # self.out_name = out_name
        self.no_simplification = no_simplification
        self.disable_batchnorm_folding = disable_batchnorm_folding
        self.keep_disconnected_nodes = keep_disconnected_nodes
        self.validation_target = validation_target
        self.strict = strict
        self.debug = debug
        self.dry_run = dry_run
        self.udo_config_paths = udo_config_paths
        self.quantization_overrides = quantization_overrides
        self.keep_quant_nodes = keep_quant_nodes
        
    def get_node_info(self,):
        onnx_mod = onnx.load(self.model_path)
        inferred_model = shape_inference.infer_shapes(onnx_mod)
        node_info = lambda info: {x.name:[y.dim_value for y in x.type.tensor_type.shape.dim] for x in info} 
        shape_info = onnx.shape_inference.infer_shapes(onnx_mod)

        inp_info = shape_info.graph.input
        out_info = shape_info.graph.output

        out_info = node_info(out_info)
        inp_info = node_info(inp_info)
        self.inp_info = inp_info
        self.out_info = out_info
        return inp_info, out_info
    
    def onnx_to_dlc(self,) -> SnpeModel:
        inp_info, out_info = self.get_node_info()    
        out_names = list(out_info.keys())
        cmd = ["snpe-onnx-to-dlc",
              "-i", self.model_path,
              "-o", self.output_path]+\
              flatten_list_args("--input_type", self.input_type)+\
              flatten_list_args("--input_dtype", self.input_dtype)+\
              flatten_list_args("--input_encoding", self.input_encoding)+\
              flatten_list_args("--input_layout", self.input_layout)+\
              make_list_args("--out_name", out_names)+\
              flatten_list_args("--udo_config_paths", self.udo_config_paths)
        
        
        if self.copyright_file:
            cmd += ["--copyright_file", self.copyright_file]
        if self.model_version:
            cmd += ["--model_version", self.model_version]
        if self.no_simplification:
            cmd += ["--no_simplification"]
        if self.disable_batchnorm_folding:
            cmd += ["--disable_batchnorm_folding"]
        if self.keep_disconnected_nodes:
            cmd += ["--keep_disconnected_nodes"]
        if self.strict:
            cmd += ["--strict"]
        if self.debug:
            cmd += ["--debug"]
        if self.dry_run:
            cmd += ["--dry_run", self.dry_run]
        if self.quantization_overrides:
            cmd += ["--quantization_overrides"]
        if self.keep_quant_nodes:
            cmd += ["--keep_quant_nodes"]
            
        subprocess.call(cmd , env=os.environ)
        
        return SnpeModel(self.output_path, inp_info, out_info)        
        
        
def flatten_list_args(args, input_args):
    ret = []
    for x in input_args:
        ret.extend([args, *x])
    return ret

def make_list_args(args, input_args):
    ret = []
    for x in input_args:
        ret.extend([args, str(x)])
    return ret


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        