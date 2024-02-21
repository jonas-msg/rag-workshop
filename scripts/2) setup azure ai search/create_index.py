from dotenv import load_dotenv, dotenv_values
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,
)
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

config = dotenv_values(".env")
service_endpoint = config.get("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = config.get("AZURE_SEARCH_INDEX_NAME")
key = config.get("AZURE_SEARCH_ADMIN_KEY")
credential = AzureKeyCredential(key)

# Configure Search Indexer
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    SearchableField(name="line", type=SearchFieldDataType.String),
    SearchableField(
        name="filename",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
    ),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="myHnswProfile",
    ),
]

# Configure Vector Search Config
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            ),
        ),
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        ),
        VectorSearchProfile(
            name="myExhaustiveKnnProfile",
            algorithm_configuration_name="myExhaustiveKnn",
        ),
    ],
)

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="line")],
        keywords_fields=[SemanticField(field_name="filename")],
    ),
)

# Create the semantic settings with config
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)
result = index_client.create_or_update_index(index)
print(f"{result.name} created")
