from selenium import webdriver
import argparse
import logging 
from retry import retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@retry(tries=50, delay=2)
def _scrape(args):
    url = f'https://gwis.jrc.ec.europa.eu/apps/gwis.statistics/seasonaltrend' 
    options = webdriver.ChromeOptions()
    preferences = {
                    "download.default_directory": args.output,
                    "directory_upgrade": True
                }

    options.add_experimental_option('prefs', preferences)
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    available_zones = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

    original_scroll_position = driver.execute_script("return window.scrollY;")
    
    for zone in available_zones:
        zone_options =  _get_elements(driver, 'zone')
        for option in zone_options:
            if option.text == zone and option.text!= 'Areas Of interest':
                logging.info(f"---------------------------------------------")
                logging.info(f"Selecting zone: {zone}")
                logging.info(f"---------------------------------------------")
                driver.execute_script("arguments[0].scrollIntoView();", option)
                option.click()
                options = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[role="listbox"] > div')))
                
                available_countries = [option.text for option in options]
                for country in available_countries:
                    for option in options:
                        if option.text == country:
                            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(option))
                            driver.execute_script("arguments[0].click();", option)
                            logging.info(f"---------------------------------------------")
                            logging.info(f"Clicked on {country}")
                            logging.info(f"---------------------------------------------")
                            driver.execute_script("window.scrollTo(0, arguments[0]);", original_scroll_position)

                            cards = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.v-card.v-sheet.theme--light')))
                            for card in cards:
                                try:
                                    element = card.find_element(By.CSS_SELECTOR, 'div.v-card__title')
                                    _download_file(driver, card, element)
        
                                except NoSuchElementException:
                                    pass

                zone_options = _get_elements(driver, 'zone')


@retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
def _download_file(driver, card, title_element):
        driver.execute_script("arguments[0].scrollIntoView(true);", card)
        driver.execute_script("window.scrollBy(0, -200);")

        download_button = WebDriverWait(card, 10).until(EC.element_to_be_clickable((By.XPATH, './/i[contains(@class, "mdi-download")]')))
        download_button.click()
        
        WebDriverWait(card, 10).until(EC.visibility_of_element_located((By.XPATH, './/a[contains(text(), "Data")]')))
        WebDriverWait(card, 10).until(EC.element_to_be_clickable((By.XPATH, './/a[contains(text(), "Data")]'))).click()
        
        WebDriverWait(card, 10).until(EC.visibility_of_element_located((By.XPATH, './/a[contains(text(), "CSV")]')))
        WebDriverWait(card, 10).until(EC.element_to_be_clickable((By.XPATH, './/a[contains(text(), "CSV")]'))).click()

        logging.info(f'{title_element.text} Download Successfully!')


@retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
def _click_selector(driver, element):
    if element == 'year':
        pos = 41
    elif element == 'zone':
        pos = 46
    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'div[aria-owns="list-{pos}"]')))
    element.click()


@retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
def _get_elements(driver, element):
    _click_selector(driver, element)
    dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.v-menu__content')))
    options = dropdown.find_elements(By.CSS_SELECTOR, 'div[role="option"]')
    return options


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?')
    args = parser.parse_args()

    return args


def main(args):
    _scrape(args)


if __name__ == "__main__":
    args = _setup_args()
    main(args)
