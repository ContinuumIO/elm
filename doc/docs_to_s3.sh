make clean html
s3cmd sync ./build/html s3://elm-docs -P
