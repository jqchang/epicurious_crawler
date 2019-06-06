import urllib3
import json
from recipe import Recipe

root_url = "https://services.epicurious.com/"
START_PAGE = 1797
PER_PAGE = 20
api_url = "api/search/v1/query?content=recipe&page={}&size={}".format(START_PAGE,PER_PAGE)
http = urllib3.PoolManager()
urllib3.disable_warnings()

all_recipes = []

if __name__ == "__main__":
    r = http.request('GET',root_url+api_url)
    ignored = 0
    errors = 0
    while r.status == 200:
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
                    all_recipes.append(ro)
                    ro.save_info()
                    print(ro.info())
                else:
                    ignored += 1
            except UnicodeEncodeError:
                ignored += 1
        if js["page"].get("nextUri"):
            api_url = js["page"]["nextUri"]
            r = http.request('GET',root_url+api_url)
        else:
            break
    print("{} recipes rejected due to errors or missing data.".format(ignored))
