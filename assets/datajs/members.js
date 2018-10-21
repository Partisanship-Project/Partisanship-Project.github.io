
function fetchMemberData(){
    var member_input=$("#member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$("#member_select").val("Select a Member");
    }else{
        console.log('got here1')
        var r=fetchMetaData(member_input);
        $('official_name').text(r['Fullname']);
        console.log(r);
    }
    
    
}
function fetchMetaData(name){
d3.csv("../assets/data/metadata.csv", function(data) {
    var row = data;
    console.log(row)
    if (row['Fullname']==name){
        return row
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