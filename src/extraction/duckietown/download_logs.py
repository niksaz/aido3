#!/usr/bin/env python

import requests
import os
from src.extraction.duckietown.config import ROOT_PAGE_URL


def download_bag_files(urls):
    # check if bag_files directory exists, else create a new one
    directory = os.path.join(os.getcwd(), "data", "bag_files")
    if not os.path.exists(directory):
        os.makedirs(directory)

    for url in urls:
        # extract bag_ID from url
        bag_ID = url

        # extract link of the bag file
        link = urls[url]

        # check that a file exists on the defined url
        response = requests.head(link)
        if response.status_code != 200:
            print("Cannot find the file {} at the link {}. Skipping it.".format(bag_ID, link))
            continue

        # define bag_name but also prevent errors for bag_ID misunderstanding (bag_ID should be without .bag extension)
        if ".bag" in bag_ID:
            bag_name = os.path.join(directory, bag_ID)
        else:
            bag_name = os.path.join(directory, bag_ID + ".bag")

        if not os.path.isfile(bag_name):
            # download file and save it to a bag file
            r = requests.get(link, allow_redirects=True)
            open(bag_name, 'wb').write(r.content)

        # print which bag files have been downloaded so far
        if ".bag" in bag_ID:
            print("The {} file is downloaded.".format(bag_ID))
        else:
            print("The {}.bag file is downloaded.".format(bag_ID))


def main():
    # insert the bag_IDs and urls of the bag files that you want to download
    # define bag_ID for better error message management
    # define full link to bag file to minimize potential link errors
    bag_ids = [
        "20171017110057_a313",
        "20171222163338_a313",
        "20171222164854_a313",
        "20171222170143_a313",
        "20171229101404_a313",
        "20171229101846_a313",
        "20171229102008_a313",
        "20171229102413_a313",
        "20171229102516_a313",
        "20171229102710_a313",
        "20171229102832_a313",
        "20171229103026_a313",
        "20171229103241_a313",
        "20171229103415_a313",
        "20171229104835_a313",
        "20171229105042_a313",
        "20171229105442_a313",
        "20171229105544_a313",
        "20171229123937_a313",
        "20171229124356_a313",
        "20171229124508_a313",
        "20171229124738_a313",
        "20171229124842_a313",
        "20171229125004_a313",
        "20171229130000_a313",
        "20171229130217_a313",
        "20171229130317_a313",
        "20171229130430_a313",
        "20171229132628_a313",
        "20171229132852_a313",
        "20171229133356_a313",
        "20180104160023_a313",
        "20180104160326_a313",
        "20180104160628_a313",
        "20180104161012_a313",
        "20180104161713_a313",
        "20180104184212_a313",
        "20180104184537_a313",
        "20180104184952_a313",
        "20180104185251_a313",
        "20180104185603_a313",
        "20180104190924_a313",
        "20180104191210_a313",
        "20180104191413_a313",
        "20180104191547_a313",
        "20180104191716_a313",
        "20180104193114_a313",
        "20180104194245_a313",
        "20180108135251_a313",
        "20180108135529_a313",
        "20180108140736_a313",
        "20180108141006_a313",
        "20180108141155_a313",
        "20180108141448_a313",
        "20180108141719_a313",
        "20180108194225_a313",
        "20180108195947_a313",
        "20180108200202_a313",
        "20180108200650_a313",
        "20180108201149_a313",
        "20180111102628_a313",
        "20180111103659_a313",
        "20180111104301_a313",
        "20180111105758_a313",
        "20180111130129_a313",
        "20180111130603_a313",
    ]
    urls = {bag_id: (ROOT_PAGE_URL + "/" + bag_id + ".bag") for bag_id in bag_ids}

    # download bag files
    download_bag_files(urls)


if __name__ == "__main__":
    main()
