from transformers import BertForQuestionAnswering, BertTokenizer
import torch

MAX_LENGTH = 512
OVERLAP = 50

weight_path = "kaporter/bert-base-uncased-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(weight_path)
model = BertForQuestionAnswering.from_pretrained(weight_path)

def create_chunks(question, context, max_length=MAX_LENGTH, overlap=OVERLAP):
    # Tokenize the question and context separately, then combine
    question_ids = tokenizer(question, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    context_ids = tokenizer(context, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    
    total_tokens = len(context_ids)
    print(f"Total tokens: {total_tokens}")

    # Calculate the space left for context tokens after the question and special tokens
    question_len = len(question_ids)
    available_context_length = max_length - question_len - 3  # Accounting for [CLS], [SEP], [SEP]
    
    chunks = []
    
    # Split the context into chunks and append the question and special tokens to each chunk
    for start in range(0, total_tokens, available_context_length - overlap):
        end = min(start + available_context_length, total_tokens)
        chunk_ids = context_ids[start:end]
        
        # Combine the question with the current chunk of context
        input_ids = torch.cat([
            torch.tensor([tokenizer.cls_token_id]),
            question_ids,
            torch.tensor([tokenizer.sep_token_id]),
            chunk_ids,
            torch.tensor([tokenizer.sep_token_id])
        ])
        
        # Create attention masks
        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        
        # Pad if necessary to match max_length
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * padding_length)])
            attention_mask = torch.cat([attention_mask, torch.tensor([0] * padding_length)])
        
        # Determine the position of the [SEP] token for token type IDs
        sep_idx = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()
        if sep_idx:
            sep_idx = sep_idx[0]
        else:
            sep_idx = len(input_ids)
        
        token_type_ids = [0] * (sep_idx + 1) + [1] * (max_length - sep_idx - 1)
        
        # Append the chunk with its corresponding attention masks and token type IDs
        chunks.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': torch.tensor(token_type_ids)
        })
        
        # For debugging, decode and print the chunk
        decoded_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"\nChunk (from token {start} to {end}):\n{decoded_chunk}")
    
    return chunks

def get_answer_from_chunks(question, context):
    chunks = create_chunks(question, context)
    best_answer = ""
    max_score = -float('inf')
    
    for i, chunk in enumerate(chunks):
        # Prepare the input tensors for the model
        input_ids = chunk['input_ids'].unsqueeze(0)
        attention_mask = chunk['attention_mask'].unsqueeze(0)
        token_type_ids = chunk['token_type_ids'].unsqueeze(0)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Get the logits for the start and end positions
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Mask the logits for special tokens (CLS, SEP, PAD)
        special_tokens_mask = (chunk['input_ids'] == tokenizer.cls_token_id) | \
                              (chunk['input_ids'] == tokenizer.sep_token_id) | \
                              (chunk['input_ids'] == tokenizer.pad_token_id)
        
        # Set special tokens' logits to negative infinity to avoid selecting them
        start_logits[special_tokens_mask] = -float('inf')
        end_logits[special_tokens_mask] = -float('inf')
        
        # Find the best answer span by checking all valid start-end pairs
        for start_idx in range(len(start_logits)):
            for end_idx in range(start_idx, len(end_logits)):  # Ensure that end_idx >= start_idx
                # Compute the score for the current span
                score = start_logits[start_idx].item() + end_logits[end_idx].item()
                
                # Update the best answer if this span's score is the highest so far
                if score > max_score:
                    max_score = score
                    answer_tokens = tokenizer.convert_ids_to_tokens(chunk['input_ids'][start_idx:end_idx + 1].tolist())
                    best_answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        # Optionally, print the best answer for each chunk
        print(f"\nChunk {i} Best Answer = {best_answer}\n")
    
    return best_answer

def main():
    # question = "What is the project sector, either Renewable Energy or Forestry and Land Use?"
    # context = "[['Social\\nWell Being', None, None, '\u2022 The project generates clean power without negative impacts\\non surroundings\\n\u2022 No human displacement due to the project activity and\\nhence no requirement of relocation\\n\u2022 The local population has been employed during the\\ninstallation, commissioning and operation of the wind mills, thus\\nproper training imparted to the people involved results in the skill\\ndevelopment of the local inhabitants and also improvement in\\ntheir economic condition.\\n\u2022 Improvement in the infrastructure in the nearby areas such\\nas development of road network, transportation facilities and\\nother amenities'], ['Economic\\nWell being', None, None, '\u2022 The project activity is responsible for creating business\\nopportunities for many local stakeholders\\n\u2022 It is an effort on the part of the project proponent to\\ncontribute towards grid stability and abridging the demand-supply\\ngap in electricity in the regional grid and in turn in the national\\ngrid\\n\u2022 The project activity conserves fossil-fuels and makes these\\nnon-renewable sources of energy available for other important\\npurposes.\\n\u2022 It indirectly contributes towards industrial development of\\nthe region by creating a support in terms of supplying power for\\nindustries to come up in due course of time'], ['', 'Technological', '', '\u2022 The project activity generates clean power by harnessing\\nthe potential wind energy for power generation'], [None, 'Well being', None, None]]\n[['Social'], ['Well Being']]\n[['Economic'], ['Well being']]\n1.16 Any information relevant for the eligibility of the project and\nquantification of emission reductions or removal enhancements, including\nlegislative, technical, economic, sectoral, social, environmental, geographic,\nsite-specific and temporal information.):\nPurpose:\nThe purpose of the wind-mills set up by the project activity is as follows:\n\u2022 Generating clean power by utilizing the renewable natural resource i.e., wind\npower and exporting the electricity generated to their own industry by making use\nof the transmission lines of TNEB. Hence the project activity does not cause\nemissions of greenhouse gases (GHGs) that would have otherwise been caused\nby power generation by the combustion of non-renewable sources of energy.\n\u2022 Harnessing the wind power potential existing in India for power generation that\nhas not been exploited to its full potential till date\n\u2022 Contribution to the industrial development of India by providing support in terms\nof enhanced power availability\n\u2022 Increasing the share of renewable energy directly in the regional electricity grid\nand indirectly in the national electricity grid\n\u2022 Contribution to the causes of fossil-fuel conservation and climate change\nmitigation\n\u2022 Contribution to nation towards energy security\n\u2022 Independency on fossil fuels\n\u2022 Saving in national revenue by avoiding import of fossil fuels\nContribution of the Project Activity to Sustainable Development:\nThe contribution of the project activity to the sustainable development of the host\ncountry India is evident from the following:\n15\n\n[['', '\u2022 The project contributes towards the stability of grid power\\nthat is a major cause of concern in remote locations\\n\u2022 It also helps in reducing the losses due to power\\ntransmission and distribution from the existing generating stations\\nof the grid to remote areas'], ['Environmental\\nWell being', '\u2022 The project activity displaces an equivalent quantum of\\npower generated by the combustion of fossil fuels, the non-\\nrenewable energy sources at the grid connected thermal power\\nplants, thus reducing GHG emissions and contributing to the\\noverall cause of mitigation of global warming\\n\u2022 The project activity by setting up wind-farms for power\\ngeneration does not cause environmental disturbance or\\necological imbalance to the surroundings\\n\u2022 The project activity does also contribute to the reduction in\\nthe levels of SOx, NOx, and SPM associated with combustion of\\nfossil fuels for generation of thermal power']]\n[['Environmental'], ['Well being']]\nVCS Project Description\nTable 1.6: Contribution to Sustainable Development"
    # question = "What are the start and end dates of the crediting period for this project?"
    # context = """
    # [['Crediting period', '', '\u2610 Seven years, twice renewable', ''], [None, None, '\u2610 Ten years, fixed', None]]\n1.10 Project Crediting Period\n9\n\n[['', None, None, '', '\u2612 Other: 40 years, renewable up to the maximum allowable', ''], [None, None, None, None, 'crediting period of 100 total years', None], ['', 'Start and end date of', '', '02-March-2024 to 01-March-2064', '02-March-2024 to 01-March-2064', None], [None, 'first or fixed crediting', None, None, None, None], [None, 'period', None, None, None, None]]\nVCS Project Description Template, v4.3
    # """
    question = "What is the methodology of this project?"
    context = "[['Source', 'Latest year for which\\nemission factor available', 'Value'], ['Third update for second NDC\\nof the UAE \u2013 2023 page 1712', '2019', '0.55 tCO \/MWh\\n2'], ['Energy profile, United Arab\\nEmirates published by IRENA\\n\u2013 8th August 2023 page 313', '2021', '457 tCO \/GWh (0.457\\n2\\ntCO \/MWh)\\n2']]\nVCS Project Description Template, v4.3\n\u2610 Yes \u2612 No\nAdditionality Methods\nAll the steps are explained in the section 3.5 additionality.\n3.6 Methodology Deviations\n1. Grid emission factor\nThe methodology refers to CDM Tool 7 for electricity grid emission factor. However, the data is\nnot available in public domain to calculate the grid emission factor and therefore project owner\nis not able to calculate this factor.\nThe project owner request to use the publicly available grid emission factor for the project,\nwhich will be fix for first crediting period. For future crediting period project owner anticipates\nthat the data will be available to calculate the grid emission factor and then the Tool 7 or any\nother Verra guidance will be used for the project.\nBased on publicly available information, the following data is available for UAE grid emission\nfactor:\nThe Project will consumed less electricity with respect to the baseline and therefore lower grid\nemission factor will result in lower baseline emissions and lower emission reductions and\ntherefore the lower grid emission factor will be conservative of therefor lower value (IRENA) is\nselected. As IRENA value is latest and conservative therefore this has been used for the first\ncrediting period.\n2. Benchmarking Boundary\n12 https:\/\/unfccc.int\/sites\/default\/files\/NDC\/2023-\n07\/Third%20Update%20of%20Second%20NDC%20for%20the%20UAE_v15.pdf (page 17)\n13 https:\/\/www.irena.org\/-\/media\/Files\/IRENA\/Agency\/Statistics\/Statistical_Profiles\/Middle-East\/United-Arab-\nEmirates_Middle-East_RE_SP.pdf (page 3)\n42\n\n[['BE\\ny', 'Baseline emissions in year y', '36,393', 'tCO e\/yr\\n2', 'Calculated']]\nVCS Project Description Template, v4.3\nAs per the Methodology the default benchmark boundary is a city where the project activity is\nlocated which is Abu Dhabi.\nNo definition is provided related with the city.\nThe UAE is divided into seven administrative divisions (Abu Dhabi, Dubai, Sharjah, Ajman, Umm\nAl Quwain, Ras Al Khaimah and Fujairah), referred as emirates. Each emirates have its own\nrules and regulations and clear demarcation of boundary.\n\uf0b7 Therefore, Abu Dhabi Emirate is considered as city for methodological requirement and\nbenchmark boundary. As this is grouped project and therefore the boundary is limited\nto Abu Dhabi Emirate, United Arab Emirates.\n4 QUANTIFICATION OF ESTIMATED GHG\nEMISSION REDUCTIONS AND\nREMOVALS\n4.1 Baseline Emissions\nBaseline emissions are estimated using Approach 1 as the benchmark installation is electricity\ndriven technology.\nTotal baseline emissions are calculated as follows, using Equation (2) from the methodological\ntool 05; Baseline, project and\/or leakage emissions from electricity consumption and\nmonitoring of electricity generation version 3.0. When applying the tool, parameter QB,y shall\ncorrespond to parameter EC whereas BE shall correspond to BE . BL,k,y, y EC,y\nEquation (1) (cid:2)(cid:3) =(cid:6) \u00d7 (cid:3)(cid:11) (cid:12) (1+(cid:16)(cid:17)(cid:18) ) (cid:4) (cid:7),(cid:4) (cid:7),(cid:4) (cid:7),(cid:4)\nWhere\nBE = Baseline emissions in year y (tCO e\/yr) y 2\nQ = Estimated electricity consumption of isolated and less efficient air- B,y\ncooled reciprocating chiller systems in year y (MWh\/yr)\nTDL = Average technical transmission and distribution losses for providing BY\nelectricity to the baseline in year y\nEF = Grid CO emission factor calculated in accordance with the \u201cTool to EL,y 2\ncalculate the emission factor for an electricity system\u201d (Version 7.0)\n(tCO e\/MWh) 2\n43"
    
    answer = get_answer_from_chunks(question, context)
    print('Predicted answer:', answer)


if __name__ == "__main__":
    main()
