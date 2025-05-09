
CREATE OR REPLACE EXTERNAL TABLE `bqml_mlb.player_images`
WITH CONNECTION `US.mlb_headshots`
OPTIONS
  ( object_metadata = 'SIMPLE',
    uris = ['gs://mlb-headshots-name-only/*']
  );


  CREATE OR REPLACE MODEL `bqml_mlb.multimodal_embedding_model`
  REMOTE WITH CONNECTION `US.mlb_headshots`
  OPTIONS (ENDPOINT = 'multimodalembedding@001');

CREATE OR REPLACE TABLE `bqml_mlb.player_image_embeddings`
AS
SELECT *
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `bqml_mlb.multimodal_embedding_model`,
    (SELECT * FROM `bqml_mlb.player_images` WHERE content_type = 'image/jpeg' LIMIT 10000))


SELECT DISTINCT(ml_generate_embedding_status),
  COUNT(uri) AS num_rows
FROM bqml_mlb.player_image_embeddings
GROUP BY 1;



CREATE OR REPLACE
  VECTOR INDEX `player_images_index`
ON
  bqml_mlb.player_image_embeddings(ml_generate_embedding_result)
  OPTIONS (
    index_type = 'IVF',
    distance_type = 'COSINE');



CREATE OR REPLACE TABLE `bqml_mlb.search_embedding`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `bqml_mlb.multimodal_embedding_model`,
  (
    SELECT '#27 Trey Sweeney (ID: 700242) | Pos: SS' AS content
  )
);


CREATE OR REPLACE TABLE `bqml_mlb.vector_search_results` AS
SELECT base.uri AS gcs_uri, distance
FROM
  VECTOR_SEARCH(
    TABLE `bqml_mlb.player_image_embeddings`,
    'ml_generate_embedding_result',
    TABLE `bqml_mlb.search_embedding`,
    'ml_generate_embedding_result',
    top_k => 5);
