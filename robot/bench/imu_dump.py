import numpy as np, sys, math
d=np.load(sys.argv[1], allow_pickle=True)
fr=d["frame"]; cmd=d["command"]; tg=d["target"]; st=d["stance"]
q=fr[:,9:21]+st; w=fr[:,21:33]/0.1
for k in range(0,len(fr),5):
    gx,gy,gz=fr[k][:3]; pitch=math.degrees(math.atan2(-gx,-gz)); roll=math.degrees(math.atan2(gy,-gz))
    err=np.degrees(np.abs(tg[k]-q[k])).max()
    print("tick %3d t %4.2f cmd %.2f pitch %+5.1f roll %+5.1f gyro_y %+5.2f  max_err %4.1f  |w|max %.1f"%(k,k*0.02,fr[k][-3],pitch,roll,fr[k][4]/0.25,err,np.abs(w[k]).max()))
