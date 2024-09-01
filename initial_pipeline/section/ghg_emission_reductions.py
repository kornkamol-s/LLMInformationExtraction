import json
import logging


def extract_data(openaiClient, text, file):
    print(text)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_ghg_emission_reductions",
                "description": "Get the ghg emissions reductions data from the section",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "year": {
                                        "type": "string",
                                        "description": "The years or duration of the record",
                                    },
                                    "estimated_ghg_emission_number": {
                                        "type": "number",
                                        "description": "Estimated GHG emission reductions or removals (tCO2e) for the year",
                                    },
                                },
                                "required": [
                                    "year",
                                    "estimated_ghg_emission_number",
                                ],
                            },
                        },
                        "total_estimated_ERs": {
                            "type": "number",
                            "description": "Total GHG emission reductions or removals (tCO2e)",
                        },
                        "total_number_of_crediting_years": {
                            "type": "number",
                            "description": "Total number of crediting years, the duratin between the first and last year of the records",
                        },
                        "average_annual_ERs": {
                            "type": "number",
                            "description": "Average GHG emission reductions or removals (tCO2e) per year",
                        },
                    },
                    "required": ["records"],
                },
            },
        }
    ]

    query = f"""Use the below article section on the {file} file to answer the subsequent question. If the answer cannot be found, write "I don't know."

    Article section:
    \"\"\"
    {text}
    \"\"\"

    Question: Get all the yearly Estimated GHG Emission Reductions or Removals records for this project, also the total estimated ERs through all the listed year, the average annual ERs, and the total number of crediting years."""

    response = openaiClient.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You answer questions about the other entities section.",
            },
            {"role": "user", "content": query},
        ],
        model="gpt-3.5-turbo",
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_ghg_emission_reductions"},
        },
        temperature=0,
    )

    if (
        response.choices[0].message.tool_calls[0].function.arguments
        and response.choices[0].message.tool_calls[0].function.name
        == "get_ghg_emission_reductions"
    ):
        logging.info("Success!")
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    else:
        logging.error("Something went wrong, function not called!")
