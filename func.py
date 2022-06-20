import io
import os
import json
import logging

import sys
from fdk import response
from apyori import apriori

import oci
import pandas as pd
import numpy as np
import time

funcDefinition = {
    "status": {
        "returnCode": 0,
        "errorMessage": ""
    },
    "funcDescription": {
        "outputs": [
            {"name": "support", "dataType": "number"},
            {"name": "confidence", "dataType": "number"},
            {"name": "lift", "dataType": "number"},
            {"name": "finalRules", "dataType": "varchar(1000)"}
        ],
        "parameters": [
            {"name": "transaction_id", "displayName": "Transaction ID",
             "description": "Choose Transaction ID Column", "required": True,
             "value": {"type": "column"}},
            {"name": "key_name", "displayName": "Key Name",
             "description": "Choose Key Name Column", "required": True,
             "value": {"type": "column"}}
        ],
        "bucketName": "bucket-FAAS",
        "isOutputJoinableWithInput": False
    }
}

def handler(ctx, data: io.BytesIO = None):
    response_data = ""
    try:
        body = json.loads(data.getvalue())
        funcMode = body.get("funcMode")
        if funcMode == 'describeFunction':
           response_data = json.dumps(funcDefinition)
        elif funcMode == "executeFunction":
            input_method = body.get("input").get("method")
            if input_method == "csv":
                bucketName = body.get("input").get("bucketName")
                fileName = body.get("input").get("fileName") + body.get("input").get("fileExtension")
                args = body.get("args")

                input_csv_path = read_from_objectstore(bucketName, fileName)
                dat = pd.read_csv(input_csv_path, sep=",", quotechar="\"", encoding="utf-8", parse_dates=True, infer_datetime_format=True)

                transactionId = args.get("transaction_id")
                keyName = args.get("key_name")

                ds_list = dat.groupby(transactionId)[keyName].apply(list)
                association_rules = list(apriori(ds_list, min_support=0.001, min_confidence=0.1, min_lift=0.5, max_length=4))
                
                supports = []
                confidences = []
                lifts = []
                bases = []
                adds = []

                for r in association_rules:
                    for s in r.ordered_statistics:
                        supports.append(r.support)
                        confidences.append(s.confidence)
                        lifts.append(s.lift)
                        bases.append(', '.join(list(s.items_base)))
                        adds.append(', '.join(list(s.items_add)))

                resultDF = pd.DataFrame({
                    'support': supports,
                    'confidence': confidences,
                    'lift': lifts,
                    'base': bases,
                    'add': adds
                })

                clean_result = resultDF.loc[resultDF['base'] != '']
                clean_result = clean_result.sort_values(by=['confidence', 'support'], ascending=[False, False])
                TopN_result = clean_result.head(25)
                Final_result = TopN_result.copy()
                Final_result.loc[:,'finalRules'] = Final_result.loc[:,'base'] + ' => '+ Final_result.loc[:,'add']

                output_dat = Final_result[['support', 'confidence', 'lift', 'finalRules']]

                outputFile = body.get("output").get("fileName") + body.get("output").get("fileExtension")
                output_csv_path  = "/tmp/"+outputFile
                output_dat.to_csv(output_csv_path, columns=["support", "confidence", "lift", "finalRules"], index=False)
                write_to_objectstore(bucketName, outputFile, output_csv_path)

                os.remove(input_csv_path)
                os.remove(output_csv_path)
                response_data = prepareResponse(bucketName, outputFile)
            else:
                response_data = prepareResponseError("input method not supported: " + input_method)
        else:
            response_data = prepareResponseError("Invalid funcMode: " + funcMode)
    except (Exception, ValueError) as ex:
        response_data = prepareResponseError("Error while executing " + ex)

    return response.Response(
        ctx, response_data,
        headers={"Content-Type": "application/json"}
    )

def prepareResponse(bucketName, outputFile):
    ret_template = """{
        "status": {
            "returnCode": "",
            "errorMessage": ""
            }
        }"""
    ret_json = json.loads(ret_template)
    ret_json["status"]["returnCode"] = 0
    ret_json["status"]["errorMessage"] = ""
    return json.dumps(ret_json)

def prepareResponseError(errorMsg):
    ret_template = """{
        "status": {
            "returnCode": "",
            "errorMessage": ""
            }
        }"""
    ret_json = json.loads(ret_template)
    ret_json["status"]["returnCode"] = -1
    ret_json["status"]["errorMessage"] = errorMsg
    return json.dumps(ret_json)

def get_object(bucketName, objectName):
    signer = oci.auth.signers.get_resource_principals_signer()
    client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    namespace = client.get_namespace().data
    try:
        print("Searching for bucket and object", flush=True)
        object = client.get_object(namespace, bucketName, objectName)
        print("found object", flush=True)
        if object.status == 200:
            print("Success: The object " + objectName + " was retrieved with the content: " + object.data.text, flush=True)
            message = object.data.text
        else:
            message = "Failed: The object " + objectName + " could not be retrieved."
    except Exception as e:
        message = "Failed: " + str(e.message)
    return { "content": message }


def read_from_objectstore(bucket_name, file_name):
    try:
        logging.getLogger().info(
            "reading from object storage {}:{}".format(bucket_name, file_name))
        signer = oci.auth.signers.get_resource_principals_signer()
        object_storage = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
        namespace = object_storage.get_namespace().data
        obj = object_storage.get_object(namespace, bucket_name, file_name)
        file = open('/tmp/'+file_name, "wb")
        for chunk in obj.data.raw.stream(2048 ** 2, decode_content=False):
            file.write(chunk)
        file.close()
        return '/tmp/'+file_name
    except Exception as e:
        print("Error found\n")
        print(e)
        return None

def write_to_objectstore(bucket_name, file_name, source_file):
    logging.getLogger().info("Writing to object storage {}:{}".format(bucket_name, file_name))
    signer = oci.auth.signers.get_resource_principals_signer()
    object_storage = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
    namespace = object_storage.get_namespace().data
    with open(source_file, 'rb') as f:
        obj = object_storage.put_object(namespace, bucket_name, file_name, f)