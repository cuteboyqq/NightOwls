import json
import os
import shutil
import numpy as np
import cv2
import glob


class NightOwls:
    def __init__(self,args):
        self.label_json = args.json
        self.save_dir = args.save_dir
        self.img_dir = args.img_dir
        self.im_width = args.im_width
        self.im_height = args.im_height
        # NightOwl label
        # 4 classes (distrbution)
        # 1:Pedestrians
        # 2:Bicycledriver
        # 3:Motorbikedriver
        # 4:Ignore areas

        # BDD100K label
        # 0: pedestrian
        # 1: rider
        # 2: car
        # 3: truck
        # 4: bus
        # 5: train
        # 6: motorcycle
        # 7: bicycle
        # 8: traffic light
        # 9: traffic sign
        # 10: stop sign
        # 11: lane marking
        self.label_mapping = {1:0, 2:1, 3:1}

    def Convert_To_YOLO_Format_Label_Txt(self):
        f= open(self.label_json)
        data = json.load(f)
        c = 1
        ## Get image id first
        for i in data["images"]:
            
            print("{}:{}".format(c,i))
            print("id:{}".format(i["id"]))
            print("image:{}".format(i["file_name"])) 
            label_txt = i["file_name"].split(".")[0] + ".txt"
            save_label_dir = os.path.join(self.save_dir,"labels")
            os.makedirs(save_label_dir,exist_ok=True)
            save_label_path = os.path.join(self.save_dir,"labels",label_txt)
            save_im_dir = os.path.join(self.save_dir,"images")
            os.makedirs(save_im_dir,exist_ok=True)
            print("save_label_path:{}".format(save_label_path))
            c+=1
            #input()
            BB_list,label_list = self.Parse_NightOwls_Annotations_v2(i["id"])
            if len(BB_list)>0:
                ## Save image.png
                image_path_list = glob.glob(os.path.join(self.img_dir,"**","*.png"))
                for im_path in image_path_list:
                    im_file = im_path.split(os.sep)[-1]
                    im_name = im_file.split(".")[0]
                    if im_file==i["file_name"]:
                        shutil.copy(im_path,save_im_dir)

                ## Save label.txt
                with open(save_label_path,'a') as f:
                    for i in range(len(BB_list)):
                        print(BB_list[i])
                        print(label_list[i])
                        la = label_list[i]
                        # if la != 4: #filter out ignore areas label
                        bdd_la = self.label_mapping[la]
                        x = float((int(((BB_list[i][0] + BB_list[i][2]/2.0) / self.im_width )*1000000))/1000000)
                        y = float((int(((BB_list[i][1] + BB_list[i][3]/2.0) / self.im_height )*1000000))/1000000)
                        w = float((int((BB_list[i][2] / self.im_width )*1000000))/1000000)
                        h = float((int((BB_list[i][3] / self.im_height )*1000000))/1000000)
                        print(f'la:{bdd_la},x:{x},y:{y},w:{w},h:{h}')
                        la_str = str(bdd_la) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                        print(la_str)
                        
                        f.write(la_str)
                        f.write("\n")
                

    def Parse_NightOwls_Images(self):
        f= open(self.label_json)
        data = json.load(f)
        c = 1
        for i in data["images"]:
            print("{}:{}".format(c,i))
            c+=1
            input()

    def Parse_NightOwls_Annotations_v2(self,id=None):
        find_id=False
        BB_list = []
        label_list = []
        f= open(self.label_json)
        data = json.load(f)
        c = 1
        for i in data["annotations"]:
            if id==i['image_id']:
                print("bbox: {}".format(i['bbox']))
                print("category_id: {}".format(i['category_id']))
                print("image_id: {}".format(i['image_id']))
                print("occluded: {}".format(i['occluded']))
                print("{}:{}".format(c,i))
                c+=1
                #input()
                find_id=True
                if i['category_id'] != 4:
                    BB_list.append(i['bbox'])
                    label_list.append(i['category_id'])
        return BB_list,label_list
            
    def Parse_NightOwls_Annotations(self):
        f= open(self.label_json)
        data = json.load(f)
        c = 1
        for i in data["annotations"]:
            print("bbox: {}".format(i['bbox']))
            print("image_id: {}".format(i['image_id']))
            print("occluded: {}".format(i['occluded']))
            print("{}:{}".format(c,i))
            c+=1
            input()

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
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="./labels_2023-11-01")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')

    parser.add_argument('-imwidth','--im-width',type=int,default=1024,help='image width')

    parser.add_argument('-imheight','--im-height',type=int,default=640,help='image height')

    return parser.parse_args()

if __name__=="__main__":
   args = get_args_label()
   nightowls = NightOwls(args)
   #nightowls.Parse_NightOwls_Annotations()
   #nightowls.Parse_NightOwls_Images()
   nightowls.Convert_To_YOLO_Format_Label_Txt()