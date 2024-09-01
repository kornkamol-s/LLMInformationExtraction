import json
import logging


def extract_data(openaiClient, text, file):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_other_entities",
                "description": "Get the other entities from the section",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "other_entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "organization_name": {
                                        "type": "string",
                                        "description": "The name of the organization.",
                                    },
                                    "role_in_project": {
                                        "type": "string",
                                        "description": "The role of the organization in the project.",
                                    },
                                    "contact_person": {
                                        "type": "string",
                                        "description": "The name of the contact person.",
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": "The title of the contact person.",
                                    },
                                    "address": {
                                        "type": "string",
                                        "description": "The address of the organization.",
                                    },
                                    "telephone": {
                                        "type": "string",
                                        "description": "The telephone number of the organization.",
                                    },
                                    "email": {
                                        "type": "string",
                                        "description": "The email of the contact person.",
                                    },
                                },
                                "required": [
                                    "organization_name",
                                    "contact_person",
                                    "title",
                                    "address",
                                    "telephone",
                                    "email",
                                ],
                            },
                        },
                    },
                    "required": ["other_entities"],
                },
            },
        }
    ]

    query = f"""Use the below article section on the {file} file to answer the subsequent question. If the answer cannot be found, write "I don't know."

    Article section:
    \"\"\"
    {text}
    \"\"\"

    Question: Get all the other entities for this project, including their organization name, role in the project if any, contact person, title, address, telephone, and email. There can be one or multiple other entities, just give me the exact information on the provided article section."""

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
            "function": {"name": "get_other_entities"},
        },
        temperature=0,
    )

    if (
        response.choices[0].message.tool_calls[0].function.arguments
        and response.choices[0].message.tool_calls[0].function.name
        == "get_other_entities"
    ):
        logging.info("Success!")
        return json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        ).get("other_entities")
    else:
        logging.error("Something went wrong, function not called!")
