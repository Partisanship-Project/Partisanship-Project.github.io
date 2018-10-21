

function getMetadata(name){
    d3.csv("../assets/data/metadata.csv", function(data) {
        if (data['Fullname']==name){
            return data
        }
    });
}

//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
var members=[];
d3.csv("../assets/data/metadata.csv", function(data) {
    members.push(data['Fullname'])
});
    return members
}