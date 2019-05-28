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
        self.imgurl = r"https://assets.epicurious.com/photos/{}/6:4/w_620%2Ch_413/{}".format(json["photoData"]["id"],json["photoData"]["filename"])
        # save image
        rc = http.request("GET",self.link)
        if(rc.status == 200):
            rc_body = str(rc.data)
            cals = re.search(r'itemprop="calories">(.+?)</span>',rc_body)
            carbs = re.search(r'itemprop="carbohydrateContent">(.+?)</span>',rc_body)
            pro = re.search(r'itemprop="proteinContent">(.+?)</span>',rc_body)
            fat = re.search(r'itemprop="fatContent">(.+?)</span>',rc_body)
            if cals and carbs and pro and fat:
                self.calories = int(cals.group(1))
                self.carbohydrates = int(carbs.group(1).split(' ')[0])
                self.protein = int(pro.group(1).split(' ')[0])
                self.fat = int(fat.group(1).split(' ')[0])

    def has_calories(self):
        return hasattr(self, 'calories')

    def info(self):
        if hasattr(self, 'calories'):
            # return (self.title,self.calories)
            return (self.title,self.calories,self.carbohydrates,self.protein,self.fat)
        else:
            return (self.title)

    def save_info(self):
        http = urllib3.PoolManager()
        r = http.request('GET', self.imgurl, preload_content=False)
        path = './images/{}'.format(self.json["photoData"]["filename"])
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname('./images/'))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(path,'wb') as out:
            out.write(r.data)
            out.close()
        with open(r'food_info.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([path,self.calories,self.carbohydrates,self.protein,self.fat])
        r.release_conn()
