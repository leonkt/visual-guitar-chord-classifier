const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

require("dotenv").config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

//USE API ENDROUTES
const networkRouter = require("./routes/network");

app.use("/network", networkRouter);

//START SERVER
app.listen(port, () => {
  console.log(`Server is running on port: ${port}`);
});
