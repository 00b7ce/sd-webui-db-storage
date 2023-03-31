import base64
from io import BytesIO
import os
import re
import modules.scripts as scripts
import gradio as gr
from pymongo import MongoClient

from modules import script_callbacks, shared

mongo_host = os.environ.get('DB_HOST', 'localhost')
mongo_port = int(os.environ.get('DB_PORT', 27017))
mongo_username = os.environ.get('DB_USER', '')
mongo_password = os.environ.get('DB_PASS', '')

creds = f"{mongo_username}:{mongo_password}@" if mongo_username and mongo_password else ""
client = MongoClient(f"mongodb://{creds}{mongo_host}:{mongo_port}/")


def get_collection():
    db = client[shared.opts.data.get("db_strage_database_name")]
    collection = db[shared.opts.data.get("db_strage_collection_name")]
    return collection

class Scripts(scripts.Script):
    def title(self):
        return "Mongo Storage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess(self, p, processed):
        collection = get_collection() if shared.opts.data.get("db_strage_is_save_db") else None
        if collection is None:
            return True
        
        info = re.findall(r"Steps:.*$", processed.info, re.M)[0]
        input_dict = dict(item.split(": ") for item in str(info).split(", "))

        if 'Face restoration' not in input_dict:
            input_dict["Face restoration"] = ''
        
        # When disable Hires. fix
        if 'Denoising strength' not in input_dict:
            input_dict["Denoising strength"] = 0
            input_dict["Hires upscale"] = 0
            input_dict["Hires steps"] = 0
            input_dict["Hires upscaler"] = ''

        for image in processed.images:
            buffer = BytesIO()
            image.save(buffer, "png")
            image_bytes = buffer.getvalue()

            collection.insert_one({
                "prompt": processed.prompt, 
                "negative_prompt": processed.negative_prompt, 
                "steps": int(input_dict["Steps"]), 
                "seed": int(processed.seed), 
                "sampler": input_dict["Sampler"],
                "Face restoration": input_dict["Face restoration"],
                "Denoising strength": float(input_dict["Denoising strength"]),
                "Hires upscale": int(input_dict["Hires upscale"]),
                "Hires steps": int(input_dict["Hires steps"]),
                "Hires upscaler": input_dict["Hires upscaler"],
                "cfg_scale": float(input_dict["CFG scale"]), 
                "model": input_dict["Model"],
                "model_hash": input_dict["Model hash"],
                "size": tuple(map(int, input_dict["Size"].split("x"))), 
                "rate": 0,
                "image": image_bytes
            })

        return True

def on_ui_settings():
    section = ('db_storage', "DB Strage")
    # Save to DB for all generated images
    shared.opts.add_option("db_strage_is_save_db", shared.OptionInfo(False, "Save to DB for all generated images", gr.Checkbox, {"interactive": True}, section=section))
    # Database Name
    shared.opts.add_option("db_strage_database_name", shared.OptionInfo('StableDiffusion', "Database Name", section=section))
    # Collection name
    shared.opts.add_option("db_strage_collection_name", shared.OptionInfo('Automatic1111', "Collection name", section=section))

script_callbacks.on_ui_settings(on_ui_settings)