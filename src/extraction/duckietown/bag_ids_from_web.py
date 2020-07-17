# Author: Mikita Sazanovich

import requests
from bs4 import BeautifulSoup
from typing import List
from src.extraction.duckietown.config import ROOT_PAGE_URL


def extract_ref_elements_for(ref_elements, duckiebot: str) -> List[List[str]]:
    bag_id_refs = []
    for ref_element in ref_elements:
        href = ref_element['href']
        if f'_{duckiebot}.bag' in href:
            bag_name = href.rsplit('/', 1)[-1]
            bag_id = bag_name.rsplit('.', 1)[0]
            bag_id_refs.append([bag_id, href])
    return bag_id_refs


def main():
    response = requests.get(ROOT_PAGE_URL, allow_redirects=True)
    if response.status_code != 200:
        print(f'Unsuccessful downloading: {response.status_code}. Finishing...')
        return 0

    # creating an object of the overridden class
    soup = BeautifulSoup(response.text, "html.parser")
    ref_elements = soup.findAll('a')

    download_server = "http://ipfs.duckietown.org:8080"
    bag_id_refs = extract_ref_elements_for(ref_elements, 'a313')
    print(len(bag_id_refs))
    for bag_id, href in bag_id_refs:
        print(f'"{bag_id}": r"{download_server}{href}",')


if __name__ == '__main__':
    main()
