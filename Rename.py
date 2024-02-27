import os
import all_actions as aa

_path = aa.data_path
# act = aa.load_all_actions()
lili = os.listdir(os.path.join(_path,"thanks"))
j = 0
for i in lili:
    pp = os.path.join(_path,"thanks",i)
    apply = os.path.join(_path,"thanks",f"thanks_{j}.jpg")
    os.rename(pp,apply)
    print(apply)
    j+=1
print("===============================")