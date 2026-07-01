import ast
import leetcode

def id_from_slug(slug: str, api_instance) -> str:
    """
    Returns the INTERNAL question ID (not the frontend/UI ID).
    This is what must be sent in the submission body to LeetCode's API.
    """
    graphql_request = leetcode.GraphqlQuery(
        query="""
            query getQuestionDetail($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    questionId
                    questionFrontendId
                }
            }
        """,
        variables={"titleSlug": slug},
        operation_name="getQuestionDetail",
    )
    response = ast.literal_eval(str(api_instance.graphql_post(body=graphql_request)))
    question = response.get('data', {}).get('question')

    if not question:
        return None

    internal_id = question.get('question_id')       # e.g. 915 — use THIS in submission body
    frontend_id = question.get('question_frontend_id')  # e.g. 478 — display only

    print(f"Frontend ID (UI): {frontend_id}, Internal DB ID: {internal_id}")
    return internal_id  # ← must return internal ID for the API


def metadata_from_slug(slug: str, api_instance) -> str:
    """
    Retrieves the metadata of the question with the given slug
    """
    graphql_request = leetcode.GraphqlQuery(
      query="""
                  query getQuestionDetail($titleSlug: String!) {
                    question(titleSlug: $titleSlug) {
                      metaData
                    }
                  }
              """,
              variables={"titleSlug": slug},
              operation_name="getQuestionDetail",
    )
    response = ast.literal_eval(str(api_instance.graphql_post(body=graphql_request)))
    metadata = response['data']['question']
    return metadata
