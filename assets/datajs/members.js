

function getMetadata(name){
    d3.csv("partisanshipproject.github.io/assets/data/metadata.csv", function(data) {
        for (i=0; i<data.length; i++){
            if (data[i]==name){
                return data[i]; //returns the row of data as {city: "seattle", state: "WA", population: 652405, land area: 83.9}
            }
        }
    });
}

//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    var members=[];
    d3.csv("partisanshipproject.github.io/assets/data/metadata.csv", function(data) {
        for (i=0; i<data.length; i++){
            members.push(data[i]['Fullname']);
        }
    });
    return members
}