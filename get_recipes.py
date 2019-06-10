import urllib3
import json
import numpy as np
import csv
from recipe import Recipe

root_url = "https://services.epicurious.com/"
START_PAGE = 1
PER_PAGE = 20
get_photo = True
api_url = "api/search/v1/query?content=recipe&page={}&size={}".format(START_PAGE,PER_PAGE)
http = urllib3.PoolManager()
urllib3.disable_warnings()

all_recipes = []
tagset = set()

if __name__ == "__main__":
    r = http.request('GET',root_url+api_url)
    ignored = 0
    get_next = True
    while get_next:
        print(r.status)
        if r.status == 200:
            j = r.data
            js = json.loads(j)
            for rc in js["items"]:
                try:
                    uses_placeholder = rc["photoData"]["filename"].startswith(r"no-recipe-card-")
                    if uses_placeholder:
                        ignored += 1
                        continue
                    ro = Recipe(rc)
                    if(ro.has_calories()):
                        if get_photo:
                            try:
                                ro.save_photo()
                            except:
                                ignored += 1
                                continue
                            ro.save_csv()
                            all_recipes.append(ro)
                            print(ro.info())
                            tagset.update(ro.tags)
                    else:
                        ignored += 1
                except UnicodeEncodeError:
                    ignored += 1
            if js["page"].get("nextUri"):
                api_url = js["page"]["nextUri"]
                r = http.request('GET',root_url+api_url)
            else:
                get_next = False
        else:
            r = http.request('GET',root_url+api_url)
    print("{} recipes found, {} recipes rejected.".format(len(all_recipes),ignored))
    taglist = list(tagset)
    print(len(taglist),"tags")
    with (open("tag_list.csv","w") as f):
        for t in taglist:
            f.write(t)
    tagvector = np.zeros((len(all_recipes),len(taglist)),dtype=np.uint8)
    for i,r in enumerate(all_recipes):
        for j,t in enumerate(taglist):
            if t in r.tags:
                tagvector[i,j] = 1
    print(tagvector)
    print(tagvector.shape)
    np.savetxt("tag_vector.csv", tagvector.astype(int), fmt="%i", delimiter=",")
