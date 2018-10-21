
function fetchMemberData(){
    var newdata=[];
    var member_input=$("#member_select").val();
    if (member_input=='' || member_input=='Select a Member'){
        member_input=$("#member_select").val("Select a Member");
    }else{
        console.log('got here1')
        newdata=fetchMetaData(member_input);
        //$('official_name').text(newdata[0].Fullname);
    }
}

function fetchMetaData(name){
    var title=''; //[Chamber] [Full Name]
    var affiliation = ''; //[party] from [State]
    var party ='';
    var state= '';
    d3.csv("../assets/data/metadata.csv", function(data) {
        console.log(data)
        if (data.Fullname==name){
            if (data.District==0){
                title='Senator ';
            } else {
                title="Representative ";
            }
            if (data.Party=='D'){
                party='Democratic Candidate for ';
            }else if (data.Party=='R'){
                party='Republican Candidate for ';
            }else{
                party='Candidate for ';
            }
            //output.push(data)
            $('#official_name').text(title.concat(data.Fullname));
            $('#affiliation').text(party.concat(data.State));
        }
    });
    //return output
}

//generates a list of members - ["John Kennedy", "John McCain"]
function getMembers(){
    var members=[];
    d3.csv("../assets/data/metadata.csv", function(data) {
        members.push(data.Fullname)
    });
    return members
}