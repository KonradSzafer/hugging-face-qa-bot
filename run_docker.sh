#!/bin/bash
docker build -t hf_qa_engine .
docker run -it hf_qa_engine bash
