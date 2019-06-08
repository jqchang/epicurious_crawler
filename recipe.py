import urllib3
import re
import csv
import os
import errno

class Recipe:
    def __init__(self,json):
        http = urllib3.PoolManager()
        urllib3.disable_warnings()
        self.json = json
        self.link = "https://www.epicurious.com/"+json["url"]
        self.title = json["hed"]
        self.imgurl = r"https://assets.epicurious.com/photos/{}/6:4/w_620%2Ch_413/{}".format(json["photoData"]["id"],json["photoData"]["filename"].replace(r' ',r'%20'))
        # save image
        rc = http.request("GET",self.link)
        if(rc.status == 200):
            rc_body = str(rc.data)
            tags = re.findall(r'"recipeCategory">(.*?)</',rc_body)
            cals = re.search(r'itemprop="calories">(.+?)</span>',rc_body)
            carbs = re.search(r'itemprop="carbohydrateContent">(.+?)</span>',rc_body)
            pro = re.search(r'itemprop="proteinContent">(.+?)</span>',rc_body)
            fat = re.search(r'itemprop="fatContent">(.+?)</span>',rc_body)
            if cals and carbs and pro and fat:
                self.calories = int(cals.group(1))
                self.carbohydrates = int(carbs.group(1).split(' ')[0])
                self.protein = int(pro.group(1).split(' ')[0])
                self.fat = int(fat.group(1).split(' ')[0])
                self.tags = tags

    def has_calories(self):
        return hasattr(self, 'calories')

    def info(self):
        if hasattr(self, 'calories'):
            return (self.title,self.calories,self.carbohydrates,self.protein,self.fat,len(self.tags))
        else:
            return (self.title)

    def save_photo(self):
        http = urllib3.PoolManager()
        r = http.request('GET', self.imgurl, preload_content=False)
        if len(r.data) < 1024:
            print("no data")
            raise Exception("Invalid image")
        imgpath = './images/{}'.format(self.json["photoData"]["filename"].replace(r' ',r'%20'))
        if not os.path.exists(os.path.dirname(imgpath)):
            try:
                os.makedirs(os.path.dirname('./images/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(imgpath,'wb') as out:
            out.write(r.data)
            out.close()
        r.release_conn()

    def save_csv(self):
        imgpath = './images/{}'.format(self.json["photoData"]["filename"].replace(r' ',r'%20'))
        with open(r'food_info.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([imgpath,self.calories,self.carbohydrates,self.protein,self.fat])
