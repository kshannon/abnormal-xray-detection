while true; do
    read -p "Create a new conda env & install project dependencies?" yn
    case $yn in
        [Yy]* ) conda env create -f environment.yml; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
