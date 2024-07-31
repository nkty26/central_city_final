import tensorflow as tf

def check_device():
    devices = tf.config.list_physical_devices()
    
    if not devices:
        print("No devices found.")
        return

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("GPU devices detected:")
        for device in gpu_devices:
            print(f"  - {device.name}")
    else:
        print("No GPU devices detected.")
    
    cpu_devices = [device for device in devices if device.device_type == 'CPU']
    if cpu_devices:
        print("CPU devices detected:")
        for device in cpu_devices:
            print(f"  - {device.name}")

if __name__ == "__main__":
    check_device()
