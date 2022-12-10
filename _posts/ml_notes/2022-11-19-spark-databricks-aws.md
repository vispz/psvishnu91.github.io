---
title: Spark on databricks with AWS
blog_type: ml_notes
excerpt: Setting up spark on databricks with an AWS backend.
layout: post
---


In this blog, I am going to play around with spark with databricks running on AWS.

### Resources and Credits
- [Abdul Zedan's wonderful YT tutorial](https://www.youtube.com/watch?v=Ocdv0Z4rwTQ)


### AWS Setup
- Setup a free trial [databricks account](https://www.databricks.com/try-databricks).<br/>
{% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/db-login.png" width="40%" %}
- Created a new S3 bucket `spark-learning` in the cheapest region Ohio.
{% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/spark-s3-bkt.png" width="40%" %}
- **Credentials configuration:** Used this [documentation](https://docs.databricks.com/administration-guide/account-api/iam-role.html) to
  * To create a new IAM role with an external AWS account ie., databricks' account (ID: 414351767826).
  * We also copy our private external account ID from <br/>
    {% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/databricks-cloud-rsrc.png" width="45%" %}
    {% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/db-external-id.png" width="45%" %}
  * Attach an inline policy. We copy the JSON for [the default policy](https://docs.databricks.com/administration-guide/account-api/iam-role.html#default-policy)
  which sets up spark in databricks' VPC instead of our private VPC.
  * Copy the ARN of the created policy back to databricks credential management.
- **Storage credentials:**
  * In the databricks storage credentials, added our bucket name and copied the generated policy.<br/>
    {% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/db-s3-storage-config.png" width="45%" %}
  * In AWS, `S3->buckets->spark-learning->Permissions->Edit bucket policy` copied over the
  JSON policy generated in databricks in the previous step.
- We are done with the AWS setup 😄

### Databricks Setup
- We create a new workspace with custom creation and apply the credentials we just generated.
- Create cluster under the create cluster tab. Settings as below
  ``` python
    # high concurrency is for a shared cluster
    Cluster Mode = Standard
    # Turn off Autopilot
    Terminate after = 15 mins
    # 10 cents an hour. Cheapest I can find.
    Worker, Driver Type = m6gd.large
    Num workers = 1
  ```
- This spun up two instances on aws.
    {% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/ec2-mc-spun-up.png" width="45%" %}
<br/>

### Add data
Let's first create a table, this can be done in the workspace console
`Data -> Table -> Create table`. For data, we are using this telecommunication data from
[kaggle](https://www.kaggle.com/datasets/abhinav89/telecom-customer). We are going
to directly upload the data from this csv file into DataBricks DBFS fmt table creator.

{% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/tbl-created.png" width="45%" %}

<br/>

### Create a notebook in databricks
We create a spark notebook with databricks.
    {% include image_no_center.html id="/assets/Images/posts/ml_notes/spark/db-create-nb.png" width="30%" %}

{% include nbviewer_gist.html src="8fc583d87c49e90dbaf7c39791594bdd" width="100%" height="450" %}
