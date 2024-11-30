import pdfplumber
import re
import pandas as pd

class PDFExtraction:
    def __init__(self, filename):
        """
        Initializes the PDFExtraction class.

        Parameters:
        - filename (str): The path to the PDF file to be processed.
        """
        self.filename = filename
        self.pdf = self._read_file()
        

    def _read_file(self):
        """
        Reads the PDF file using pdfplumber and returns the PDF object.

        Returns:
        - The opened PDF object.
        """
        return pdfplumber.open(self.filename)


    def _get_toc(self):
        """
        Extracts the list of headings from the PDF based on a predefined pattern.

        Returns:
        - DataFrame: A DataFrame containing the sections and their corresponding page ranges.

        This code was inspired by the concepts and evolved from the preliminary framework, which is referenced in Appendix 5 of the main report.
        """
        toc = []

        # Define a pattern to match section headings in the TOC
        # This will capture digits(1-9) or alphabets, followed with dot and digits such as 1.1 or A.1
        # Then, optionally follow with more dot and digits (1.2.1), or only dot at the end (1.2.)
        # Finally, pattern required to be followed with whitespace and uppercase letter
        pattern = re.compile(r"(?:[1-9]|[a-zA-Z])\.\d+(?:\.\d+|\.|\.\d+\.)?\s+[A-Z]+")

        # This will exclude pages containing 10 consecutive dots, dashes, or slashes, which typically indicate a table of contents page to avoid duplicating headings collected from the document's content.
        exclude_pattern = re.compile(r'[\.\-\_]{10,}')

        # Remove false headings like "2.1 MW" to ensure only valid headings are retained.
        exclude_space_mw_pattern = re.compile(r'\sMW\s')

        # Iterately extract content from document's pages
        for i, page in enumerate(self.pdf.pages):
            # Extract text from the page, removing duplicate characters and accounting for text layout
            text = page.dedupe_chars().extract_text(x_tolerance=1, y_tolerance=3)
            # Filter out table of contents pages
            if text and 'table of contents' not in text.lower() and \
                not exclude_pattern.search(text):
                # Collect line matched with defined patterns
                toc.extend((line, i) for line in text.splitlines() if pattern.match(line) and not exclude_pattern.search(line) and not exclude_space_mw_pattern.search(line))
            page.flush_cache()  

        # Filter out invalid headings
        final_toc = self._filter_toc(toc)

        # Define dataframe with heading and corresponding start page
        df = pd.DataFrame([{k: v for d in final_toc for k, v in d.items()}]).T
        df.reset_index(inplace=True)
        df.columns = ['section', 'start_page']
        df['start_page'] = df['start_page'].astype('int')

        # Get end page from the next heading's start page
        df['end_page'] = df['start_page'].shift(-1).fillna(df['start_page']).astype('int')

        return df
    

    def _filter_toc(self, toc):
        """
        Filters all possible headings based on pattern to create a structured list of sections 
        while maintaining their hierarchical relationships.

        Parameters:
            toc (list of tuples): A list containing tuples where each tuple consists of:
                - header (str): The title of the section.
                - page (int): The page number where the section starts.

        Returns:
            list of dict: A structured list of dictionaries, where each dictionary contains:
                - header (str): The title of the section.
                - page (int): The page number where the section starts.
        """
        final_toc = []
        previous_sections = []
        
        # Iterate through each header
        for header, page in toc:

            # Get the section number before first whitespace character (such as "1.1" from "1.1 Project Location")
            section = header.split(' ')[0]

            # Get all section orders such as "1.12.3 Project Description" to be [1, 12, 3]
            section_parts = [s for s in section.split('.') if s != '']
            
            # Iterate through each level to convert all level to numeric
            for i, part in enumerate(section_parts):
                if part.isalpha():
                    # Convert alphabet to number such as "A" from "A.3.5" to [1, 3, 5]
                    section_parts[i] = ord(part.lower()) - ord('a') + 1
                else:
                    # Convert number to integer type
                    section_parts[i] = int(part)
            
            # If this is the first header, append into the final list
            if not previous_sections:
                final_toc.append({header.strip(): page})
                previous_sections = section_parts
                continue
            
            is_continuation = False
            
            # Iterate through each level to validate hierarchical relationships
            for level in range(min(len(section_parts), len(previous_sections))):
                # Compare the current section number with the previous section number
                # This ensures that the current header maintains the correct hierarchy relative to the previous one.

                if section_parts[level] == previous_sections[level]:
                    # If the current section number is the same as the previous section number at this level, it indicates that they are at the same hierarchical level, so it continues checking deeper levels.
                    continue
                elif section_parts[level] == previous_sections[level] + 1:
                    # If the current section number is one greater than the previous section number, it indicates that the current header is a continuation of the previous one at this level (e.g., from 1.12.1 to 1.12.2).
                    is_continuation = True
                    #  Exit the loop as the hierarchical relationship is valid
                    break
                else:
                    # If neither of the above conditions are met, it indicates a break in the hierarchy.
                    is_continuation = False
                    #  Exit the loop as the hierarchical relationship is not valid
                    break
            
            # Validate the case where the current section is a sub-section of the previous section.
            # For example, if the current section is "1.3.1" and the previous section is "1.3", this relationship is valid.
            if len(previous_sections) and section_parts[0] <= previous_sections[0]+1:
                is_continuation = True

            # Append valid headers into final list
            if is_continuation:
                final_toc.append({header.strip(): page})
                previous_sections = section_parts

        return final_toc


    def _extract_page_range(self, start, end, start_keyword, end_keyword):
        """
        Extracts text from a range of pages between specified start and end keywords.

        Parameters:
        - start (int): The starting page number.
        - end (int): The ending page number.
        - start_keyword (str): The keyword marking the start of extraction.
        - end_keyword (str): The keyword marking the end of extraction.

        Returns:
        - The extracted text from the page range.

        This code was inspired by the concepts and evolved from the preliminary framework, which is referenced in Appendix 5 of the main report.
        """
        texts = ''

        # Iteratively extract pages from given ranges
        for i in range(start, end+1):
            if start == end:
                # Filter out the content before and after the focusing section
                cropped = self._crop_page(self.pdf.pages[i], start_keyword, end_keyword)
            elif i == start:
                # Filter out the content before the focusing starting keywords
                cropped = self._crop_page(self.pdf.pages[i], start_keyword, '')
            elif i == end:
                # Filter out the content after the focusing section using end keywords
                cropped = self._crop_page(self.pdf.pages[i], '', end_keyword)
            else:
                # Extract whole page between start and end pages
                cropped = self.pdf.pages[i]

            # Fitler out the end keywords from the selected content
            text = self._extract_page(cropped).replace(end_keyword, '')
            texts = texts + '\n' + text
        return texts


    def _crop_page(self, page, start_keyword, end_keyword):
        """
        Crops the page based on specified start and end keywords.

        Parameters:
        - page (pdfplumber object): The page object initilized using pdfplumber.
        - start_keyword (str): The keyword marking the top of the crop.
        - end_keyword (str): The keyword marking the bottom of the crop.

        Returns:
        - cropped_page (pdfplumber object): The cropped page object.
        """
        start_coords = (0, 0)
        end_coords = (page.width, page.height)
        
        if start_keyword:
            # Search and get vertical position of start keyword in a page
            start_results = page.search(start_keyword, regex=False, case=False, return_groups=True, return_chars=True)
            if start_results:
                start_coords = (0, start_results[0]['top'])

        if end_keyword:
            # Search and get vertical position of end keyword in a page
            end_results = page.search(end_keyword, regex=False, case=True, return_groups=True, return_chars=True)
            if end_results:
                end_coords = (page.width, end_results[0]['bottom'])

        # Crop the page based on start and end position
        crop_box = (0, start_coords[1], page.width, end_coords[1])
        cropped_page = page.within_bbox(crop_box, strict=False)
        return cropped_page


    def _extract_page(self, page):
        """
        Extracts both text and tables from the provided page.

        Parameters:
        - page (pdfplumber object): The page object initilized using pdfplumber.

        Returns:
        - The combined text and table content extracted from the page.

        Code updated from : https://stackoverflow.com/questions/71612119/how-to-extract-texts-and-tables-pdfplumber
        """

        # Find possible tables from provided page
        tables = page.find_tables()

        # Get table's coordinates
        table_borders = [table.bbox for table in tables]

        # Extract the content of each table and its vertical position on the page
        extracted_tables = [{'table': table.extract(), 'doctop': table.bbox[1]} for table in tables]

        # Extract all words from the page, removing duplicates
        all_words = page.dedupe_chars().extract_words()

        # Filter out words that are within the areas of the tables
        filtered_words = [word for word in all_words if not any(self._is_word_within_table_area(word, bbox) for bbox in table_borders)]
        final_text = ''
        
        # Cluster words and extracted tables based on their vertical position to keep the original layout
        for cluster in pdfplumber.utils.cluster_objects(filtered_words + extracted_tables, 'doctop', tolerance=5):
            if 'text' in cluster[0]:
                final_text += '\n' + ' '.join(item['text'] for item in cluster)
            elif 'table' in cluster[0]:
                final_text += '\n' + str(cluster[0]['table'])
        return final_text


    def _is_word_within_table_area(self, word, table_area):
        """
        Checks if a word is located within a table's area.

        Parameters:
        - word (dict): The word with its coordinates.
        - table_area (tuple): The coordinates of the table.

        Returns:
        - bool: True if the word is within the table's area, False otherwise.

        Code updated from : https://stackoverflow.com/questions/71612119/how-to-extract-texts-and-tables-pdfplumber
        """
        word_area = (word['x0'], word['top'], word['x1'], word['bottom'])
        return (word_area[0] > table_area[0] and
                word_area[1] > table_area[1] and
                word_area[2] < table_area[2] and
                word_area[3] < table_area[3])
    

    def _search_keywords(self, keys):
        """
        Searches for specified keywords within the PDF document.

        Parameters:
            keys (list): A list of keywords to search for in the PDF.

        Returns:
            A dictionary where each key is a keyword and the value is a list of page numbers containing that keyword.
        """        
        results = {key: [] for key in keys}

        # Iterate through all the pages
        for _, page in enumerate(self.pdf.pages):

            # Extract text from the current page
            text = page.dedupe_chars().extract_text()

            # Check for each keyword
            for key in keys:
                pattern = re.escape(key) if '|' not in key else key

                # Search for the pattern in the text
                if re.search(pattern, text, re.IGNORECASE):

                    # Append the page number to the result list for that keyword
                    results[key].append(page.page_number)
            page.flush_cache()   

        return results