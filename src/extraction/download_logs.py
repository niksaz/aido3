#!/usr/bin/env python

import requests
import os

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
            raise RuntimeError("Cannot find the file {} at the link {}".format(bag_ID, link))

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
    urls = {
        # "bag_ID" : "full link to bag file"
        "20180108135529_a313": r"http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20180108135529_a313.bag",
        "20180108141006_a313": r"http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20180108141006_a313.bag",
        "20180108141155_a313": r"http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20180108141155_a313.bag",
        "20180108141448_a313": r"http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20180108141448_a313.bag",
        "20180108141719_a313": r"http://ipfs.duckietown.org:8080/ipfs/QmUbtwQ3QZKmmz5qTjKM3z8LJjsrKBWLUnnzoE5L4M7y7J/logs/20180108141719_a313.bag"
    }

    # download bag files
    download_bag_files(urls)

if __name__ == "__main__":
    main()
