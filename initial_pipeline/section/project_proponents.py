import json
import logging


def extract_data(openaiClient, text, file):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_project_proponents",
                "description": "Get the project proponents from the text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_proponents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "organization_name": {
                                        "type": "string",
                                        "description": "The name of the organization.",
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
                    "required": ["project_proponents"],
                },
            },
        }
    ]

    query = f"""Use the below article section on the {file} file to answer the subsequent question. If the answer cannot be found, write "I don't know."

    Article section:
    \"\"\"
    {text}
    \"\"\"

    Question: Get all the project proponents for this project, including their organization name, contact person, title, address, telephone, and email."""

    response = openaiClient.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You answer questions about the project proponent section.",
            },
            {"role": "user", "content": query},
        ],
        model="gpt-3.5-turbo",
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_project_proponents"},
        },
        temperature=0,
    )

    if (
        response.choices[0].message.tool_calls[0].function.arguments
        and response.choices[0].message.tool_calls[0].function.name
        == "get_project_proponents"
    ):
        logging.info("Success!")
        return json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        ).get("project_proponents")
    else:
        logging.error("Something went wrong, function not called!")
        return None
