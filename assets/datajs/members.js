
function fetchMemberData(){
    var newdata=[];
    var member_input=$("#member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$("#member_select").val("Select a Member");
    }else{
        console.log('got here1')
        newdata=fetchMetaData(member_input);
        $('official_name').text(newdata[0]['Fullname']);
    }
}

function fetchMetaData(name){
    var output=[];
    d3.csv("../assets/data/metadata.csv", function(data) {
        var row = data;
        console.log(row)
        if (row['Fullname']==name){
            console.log('got here2')
            output.push(row) 
        }
    });
    return output
}

//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    var members=[];
    d3.csv("../assets/data/metadata.csv", function(data) {
        members.push(data['Fullname'])
    });
    return members
}