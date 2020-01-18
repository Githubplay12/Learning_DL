import requests
from bs4 import BeautifulSoup
import os
import shutil
from requests.exceptions import ConnectionError, InvalidSchema, MissingSchema
import time

panda_path = r'C:\Users\CARBON\Desktop\Learning\DL For Computer Vision\datasets\Catdogs\Panda'

page = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02510455')
soup = BeautifulSoup(page.content, 'html.parser')
nb_files = len(os.listdir(panda_path))
i = nb_files + 3

for line in soup.text.splitlines()[i:]:
    try:
        r = requests.get(line, stream=True)
        if r.status_code == 200:
            with open(os.path.join(panda_path, str(i) + '.jpg'), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
                print(f'Downloaded picture {str(i)}')
                i += 1
    except ConnectionError as e:
        print(e.__class__)
        continue
    except InvalidSchema as e:
        print(e.__class__)
        continue
    except MissingSchema as e:
        print(e.__class__)
        continue
    except Exception as e:
        print(e.__class__)
        continue

    if i == 3011:
        print('Done')
        break
