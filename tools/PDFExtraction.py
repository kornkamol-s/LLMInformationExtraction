import pdfplumber
import re
import pandas as pd

class PDFExtraction:
    def __init__(self, filename):
        self.filename = filename
        self.pdf = self._read_file()
        
    def _read_file(self):
        return pdfplumber.open(self.filename)

    def _get_toc(self):
        toc = []
        final_toc = []
        # pattern = `re.compile(r"(?:[1-9]|[a-zA-Z])\.\d+(?:\.\d+|\.)?\s+[A-Z]+")`
        # pattern = re.compile(r"(?:[1-9]|[a-zA-Z])\.\d+(?:\.\d+|\.|\.\d+\.|\.\d+\.\d+\.|\.\d+\.\d+)?\s+[A-Z]+")
        pattern = re.compile(r"(?:[1-9]|[a-zA-Z])\.\d+(?:\.\d+|\.|\.\d+\.)?\s+[A-Z]+")
        exclude_pattern = re.compile(r'[\.\-\_]{10,}')
        exclude_space_mw_pattern = re.compile(r'\sMW\s')

        for i, page in enumerate(self.pdf.pages):
            text = page.dedupe_chars().extract_text(x_tolerance=1, y_tolerance=3)
            print(text)
            if text and 'table of contents' not in text.lower() and not exclude_pattern.search(text):
                toc.extend((line, i) for line in text.splitlines() if pattern.match(line) and not exclude_pattern.search(line) and not exclude_space_mw_pattern.search(line))
            page.flush_cache()  
        
        print(toc)
        previous_sections = []

        for header, page in toc:
            section = header.split(' ')[0]
            section_parts = [s for s in section.split('.') if s != '']
            for i, part in enumerate(section_parts):
                if part.isalpha():
                    section_parts[i] = ord(part.lower()) - ord('a') + 1
                else:
                    section_parts[i] = int(part)
            
            if not previous_sections:
                final_toc.append({header.strip(): page})
                previous_sections = section_parts
                continue
            
            is_continuation = False
            
            for level in range(min(len(section_parts), len(previous_sections))):
                if section_parts[level] == previous_sections[level]:
                    continue
                elif section_parts[level] == previous_sections[level] + 1:
                    is_continuation = True
                    break
                else:
                    is_continuation = False
                    break
            
            if is_continuation or (len(section_parts) > len(previous_sections) and section_parts[0] <= previous_sections[0]+1):
                final_toc.append({header.strip(): page})
                previous_sections = section_parts

        df = pd.DataFrame([{k: v for d in final_toc for k, v in d.items()}]).T
        df.reset_index(inplace=True)
        df.columns = ['section', 'start_page']
        df['start_page'] = df['start_page'].astype('int')
        df['end_page'] = df['start_page'].shift(-1).fillna(df['start_page']).astype('int')

        return df


    def _search_keywords(self, keys):
        results = {key: [] for key in keys}
        for i, page in enumerate(self.pdf.pages):
            text = page.dedupe_chars().extract_text()

            for key in keys:
                pattern = re.escape(key) if '|' not in key else key
                if re.search(pattern, text, re.IGNORECASE):
                    results[key].append(page.page_number)

            page.flush_cache()   
        return results
    

    def _extract_page_range(self, start, end, start_keyword, end_keyword):
        texts = ''
        for i in range(start, end+1):
            if start == end:
                cropped = self._crop_page(self.pdf.pages[i], start_keyword, end_keyword)
            elif i == start:
                cropped = self._crop_page(self.pdf.pages[i], start_keyword, '')
            elif i == end:
                cropped = self._crop_page(self.pdf.pages[i], '', end_keyword)
            else:
                cropped = self.pdf.pages[i]
            text = self._extract_page(cropped).replace(end_keyword, '')
            texts = texts + '\n' + text
        return texts


    def _crop_page(self, page, start_keyword, end_keyword):
        start_coords = (0, 0)
        end_coords = (page.width, page.height)
        
        if start_keyword:
            start_results = page.search(start_keyword, regex=False, case=False, return_groups=True, return_chars=True)
            if start_results:
                start_coords = (0, start_results[0]['top'])

        if end_keyword:
            end_results = page.search(end_keyword, regex=False, case=True, return_groups=True, return_chars=True)
            if end_results:
                end_coords = (page.width, end_results[0]['bottom'])

        crop_box = (0, start_coords[1], page.width, end_coords[1])
        cropped_page = page.within_bbox(crop_box, strict=False)
        return cropped_page


    def _extract_page(self, page):
        # Reference : https://stackoverflow.com/questions/71612119/how-to-extract-texts-and-tables-pdfplumber
        tables = page.find_tables()
        table_borders = [table.bbox for table in tables]
        extracted_tables = [{'table': table.extract(), 'doctop': table.bbox[1]} for table in tables]
        all_words = page.dedupe_chars().extract_words()
        non_table_words = [word for word in all_words if not any(self._is_word_within_bbox(word, bbox) for bbox in table_borders)]
        combined_text_and_tables = ''
        for cluster in pdfplumber.utils.cluster_objects(non_table_words + extracted_tables, 'doctop', tolerance=5):
            if 'text' in cluster[0]:
                combined_text_and_tables += '\n' + ' '.join(item['text'] for item in cluster)
            elif 'table' in cluster[0]:
                combined_text_and_tables += '\n' + str(cluster[0]['table'])
        return combined_text_and_tables


    def _is_word_within_bbox(self, word, table_bbox):
        # Reference : https://stackoverflow.com/questions/71612119/how-to-extract-texts-and-tables-pdfplumber
        word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
        return (word_bbox[0] > table_bbox[0] and
                word_bbox[1] > table_bbox[1] and
                word_bbox[2] < table_bbox[2] and
                word_bbox[3] < table_bbox[3])