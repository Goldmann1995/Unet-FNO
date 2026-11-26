from data_training import UNetFNO2d
from compare_models import UFNO2d, FNO2d
from ufno  import Net2d
m1 = Net2d(modes1=8, modes2=8, width=32)
m2 = UNetFNO2d(modes1=8, modes2=8, width=32)
m3 = FNO2d(modes1=8, modes2=8, width=32)
print(f"FNO:      {m3.count_params():,}")
print(f"U-FNO:    {m1.count_params():,}")
print(f"UNet-FNO: {m2.count_params():,}")
