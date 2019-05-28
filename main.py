import urllib3
import json
from recipe import Recipe

root_url = "https://services.epicurious.com/"
api_url = "api/search/v1/query?content=recipe"
http = urllib3.PoolManager()
urllib3.disable_warnings()

all_recipes = []

if __name__ == "__main__":
    r = http.request('GET',root_url+api_url)
    no_cal = 0
    while r.status == 200:
        j = r.data
        js = json.loads(j)
        for rc in js["items"]:
            ro = Recipe(rc)
            if(ro.has_calories()):
                all_recipes.append(ro)
                ro.save_info()
                print(ro.info())
            else:
                no_cal += 1

        api_url = js["page"]["nextUri"]
        r = http.request('GET',root_url+api_url)
    print("{} recipes found.")
    print("{} recipes rejected due to no calories.")
