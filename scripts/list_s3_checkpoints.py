"""List S3 checkpoints for our best MLflow runs."""
import boto3

s3 = boto3.client("s3", endpoint_url="http://207.180.243.242:9000")

runs = [
    ("rootrel-wipose", "mlflow/4/be52da0cebdb4e8e9c6cfa33bda591ca/artifacts/"),
    ("metafi-mmfi", "mlflow/4/4880a549c9494b3999052afb1300d3f2/artifacts/"),
    ("mmfi-only", "mlflow/4/dc309454d4504b72b8118bd80a1dcbbb/artifacts/"),
    ("rootrel-combined", "mlflow/4/54d03a07d592476f8863b9636c680e26/artifacts/"),
]

for name, prefix in runs:
    print(f"=== {name} ===")
    resp = s3.list_objects_v2(Bucket="rfpose", Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        size = obj["Size"]
        print(f"  {key}  ({size:,} bytes)")
    print()
