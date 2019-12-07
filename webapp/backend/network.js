const router = require("express").Router();

import * as cocoSsd from "@tensorflow-models/coco-ssd";

import image from "../../src/assets/cat.jpg";


router.route("/").get((req, res) => {
  const model = await cocoSsd.load()
  const Predictions = await model.detect(image);

  console.log(Predictions)
});

module.exports = router;
