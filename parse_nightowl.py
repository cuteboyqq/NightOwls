import json
import os
import shutil
import numpy as np
import cv2



class NightOwls:
    def __init__(self,args):
        self.label_json = args.json



    def Parse_NightOwls(self):
        f= open(self.label_json)
        data = json.load(f)
        # c = 1
        # for i in data["annotations"]:
        #     print("{}:{}".format(c,i))
        #     c+=1

        print(json.dumps(data, indent = 4, sort_keys=True))

    def Show_Label_Json(self):
        with open(self.label_json) as f:
            d = json.load(f)
            print(d)
    
    def Show_Label_Json_Readable(self):
        f= open(self.label_json)
        data = json.load(f)
        # c = 1
        # for i in data["annotations"]:
        #     print("{}:{}".format(c,i))
        #     c+=1
        print(json.dumps(data, indent = 4, sort_keys=True))


    

def get_args_label():
    import argparse
    parser = argparse.ArgumentParser()
  
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/NightOwls/images")
    parser.add_argument('-json','--json',help='json label path',default="/home/ali/Projects/datasets/NightOwls/labels/nightowls_training.json")

    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="../tools/nuImages-Lane-Erode/")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')

    return parser.parse_args()

if __name__=="__main__":
   args = get_args_label()
   nightowls = NightOwls(args)
   nightowls.Show_Label_Json_Readable()