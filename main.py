from lvlm_tracker.scrape_mathvista import scrape_mathvista
from lvlm_tracker.scrape_mmmu import scrape_mmmu

def main():
    scrape_mmmu()
    scrape_mathvista()

if __name__ == '__main__':
    main()