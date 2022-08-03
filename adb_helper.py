from adb_shell.adb_device import AdbDeviceTcp, AdbDeviceUsb
from adb_shell.auth.sign_pythonrsa import PythonRSASigner

class ADB():
    def __init__(self, root: str):
        """
        Args:
        root:
            Root path of adbkey, usually is "/home/xxx/./android"
        """
        with open(root + "adbkey.pub", "r") as f:
            pub = f.read()
        with open(root + "/adbkey", "r") as f:
            priv = f.read()
        self.device = AdbDeviceUsb()
        self.signer = PythonRSASigner(pub, priv)
        
    def push(self, src: str, dst: str, **kwargs):
        """
        Args:
        src: 
            path of file in host device
        dst: 
            path of file in mobile device
        """
        try:
            self.device.connect(rsa_keys=[self.signer])
            ret = self.device.push(src, dst, **kwargs)
            return ret
            self.device.close()
        except Exception as e:
            print(f"Error in push cmd from {src} to {dst}: {e}")
        finally:
            self.device.close()    
            
    def pull(self, src: str, dst: str, **kwargs):
        """
        Args:
        src: 
            path of file in mobile device
        dst: 
            path of file in host device
        """
        try:
            self.device.connect(rsa_keys=[self.signer])
            ret = self.device.pull(src, dst, **kwargs)
            return ret
            self.device.close()
        except Exception as e:
            print(f"Error in pull cmd from {src} to {dst}: {e}")
        finally:
            self.device.close()
            
            
    def shell(self, cmd: str, **kwargs):
        """
        Args:
        cmd: adb command
        """
        try:
            self.device.connect(rsa_keys=[self.signer])
            ret = self.device.shell(cmd, **kwargs)
            return ret
        except Exception as e:
            print(f"Error in shell cmd {cmd}: {e}")
        finally:
            self.device.close()