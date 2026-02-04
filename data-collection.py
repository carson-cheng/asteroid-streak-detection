import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
a, b, wcs = None, None, None
year = 2021
def get_asteroid_image(obj_name, date, fracdays, filename=None, idx="auto", style="diff"):
    global a, b, wcs
    # date: (year, month, day)
    year, month, day = date
    if filename == None and fracdays != []:
        print("Downloading file...")
        # below logic will fail for last month of day (forced +1 for end date)
        cmd = f'''wget "https://irsa.ipac.caltech.edu/cgi-bin/MOST/nph-most?catalog=ztf&input_type=name_input&obj_name={obj_name}&obs_begin={year}+{month:02}+{day:02}&obs_end={year}+{month:02}+{(day + 1):02}&output_mode=Brief" -O out.tbl'''
        print(cmd)
        os.system(cmd)
    match = 0
    if idx is not None:
        # start at 20
        for i in range(20, len(open("out.tbl").read().split("\n"))-1):
            #print(i)
            print(len(open("out.tbl").read().split("\n")))
            a = open("out.tbl").read().split("\n")[i].split()
            a.pop(23)
            b = open("out.tbl").read().split("\n")[18].split("|")[1:-1]
            assert len(a) == len(b)
            b = [x.strip() for x in b]
            idx_ffd = b.index('filefracday')
            idx_qid = b.index('qid')
            idx_ccdid = b.index('ccdid')
            idx_field = b.index('field')
            idx_fc = b.index('filtercode')
            idx_itc = b.index('imgtypecode')
            ra = b.index('ra_obj')
            dec = b.index('dec_obj')
            yyyy = a[idx_ffd][:4]
            mmdd = a[idx_ffd][4:8]
            fracday = a[idx_ffd][8:]
            #print(fracday)
            #if int(fracday) in fracdays:
            if (idx == 'auto' and int(fracday) in fracdays) or (i == idx and idx != 'auto'):
                match += 1
                filtercode = a[idx_fc]
                field = a[idx_field]
                ccdid = a[idx_ccdid]
                qid = a[idx_qid]
                filefracday = a[idx_ffd]
                imgtypecode = a[idx_itc]
                url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{yyyy}/{mmdd}/{fracday}/ztf_{filefracday}_000{field}_{filtercode}_c{int(ccdid):02}_{imgtypecode}_q{qid}_scimrefdiffimg.fits.fz"
                if style == "science":
                    url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{yyyy}/{mmdd}/{fracday}/ztf_{filefracday}_000{field}_{filtercode}_c{int(ccdid):02}_{imgtypecode}_q{qid}_sciimg.fits"
                #print(url)
                fn = "diffimg.fits.fz"
                cmd = f"wget {url} -O {fn}"
                os.system(cmd)
                break
    if match == 0:
        print("Warning: no matches found. Returning dummy values.")
        return None, None
    fn = "diffimg.fits.fz"
    hdul = fits.open(fn)
    img =None
    if style == "science":
        img = hdul[0].data
    else:
        img = hdul[1].data
    # Create a copy, replace NaN with a fixed value
    img_display = img.copy()
    img_display[np.isnan(img)] = 0  # or np.nanmedian(img[~np.isnan(img)])
    ra_obj, dec_obj = a[ra], a[dec]
    wcs = None
    if style == "science":
        wcs = WCS(hdul[0].header)
    else:
        wcs = WCS(hdul[1].header)
    x_pixel, y_pixel = wcs.wcs_world2pix(float(ra_obj), float(dec_obj), 0)
    return img_display, (x_pixel - 1, y_pixel - 1)
#image, coords = get_asteroid_image("1862")
import os, requests
import joblib
filenames = ['Apollos.txt', 'Amors.txt', 'Atens.txt']
asteroids = []
for file in filenames:
    lns = open(file).read().split("\n")
    asteroids.extend([x.split()[0] + " " + x.split()[1] for x in lns if (len(x.split()) > 0 and x.split()[0] == '2022')])
for asteroid in asteroids:
    try:
        print(asteroid)
        #asteroid = "2021 SG"
        #cmd = f"wget https://minorplanetcenter.net/tmp2/{asteroid}.txt -O info.txt"
        #print(cmd)
        #os.system(cmd)
        #lines = open("info.txt").read().split("\n")
        import requests
        import json
        response = requests.get("https://data.minorplanetcenter.net/api/get-obs", json={"desigs": [asteroid], "output_format": ["OBS80"]})
        lines = []
        if response.ok:
            print("response returned ok")
            obs80_string = response.json()[0]['OBS80']
            lines = obs80_string.split("\n")
            #print(obs80_string)
        else:
            print("Error: ", response.status_code, response.content)
        year, month, day = 0, 0, 0
        fracdays = []
        #print(response)
        for line in lines:
            fields = line.split()
            if len(fields) > 0 and "I41" in fields[-1]: # ZTF data
                try:
                    magn = float(fields[-1][:5])
                except:
                    magn = float(fields[-2])
                if magn < 20:
                    if int(fields[1][-4:]) < 2022:
                        year = int(fields[1][-4:])
                        month = int(fields[2])
                        day = int(fields[3][:2])
                        fracday = int(fields[3][3:9])
                        fracdays.append(fracday)
        date = (year, month, day)
        print(fracdays)
        image, coords = get_asteroid_image(asteroid.replace("_", " "), date, fracdays, idx=20)
        if type(image) != type(None):
            print(type(image))
            print("Saving to file")
            joblib.dump((image, coords), f'2022/{asteroid}.joblib')
    except Exception as e:
        print("Skipping asteroid due to error: ")
        print(e)