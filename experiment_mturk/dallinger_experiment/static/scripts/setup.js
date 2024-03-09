let curr_generation, generation_size, read_multiple_versions, num_stories_to_read;

function fail(rejection){
    // A 403 is our signal that it's time to go to the questionnaire
    if (rejection.status === 403) {
      dallinger.allowExit();
      dallinger.goToPage('questionnaire');
    } else {
      dallinger.error(rejection);
    }
}

  // Create the agent.
function createParticipantAndAgent() {
    dallinger.createParticipant()
        .done(function(){
            dallinger.createAgent()
            .done(function (resp) {
                sessionStorage.setItem('node_id',resp.node.id);
                curr_generation = +resp.node.property3;
                sessionStorage.setItem('generation',curr_generation);
                sessionStorage.setItem('replication',resp.node.property5);
                console.log(resp.node);
                dallinger.getExperimentProperty('generation_size')
                    .done(function (propertiesResp) {
                        generation_size = +propertiesResp.generation_size;
                        sessionStorage.setItem('generation_size',generation_size);
                        dallinger.getExperimentProperty('read_multiple_versions')
                        .done(function(propertiesResp){
                            read_multiple_versions = propertiesResp.read_multiple_versions;
                            sessionStorage.setItem('read_multiple_versions',read_multiple_versions);
                            num_stories_to_read = (curr_generation==0 && read_multiple_versions==0) ? 1: generation_size;
                            sessionStorage.setItem('num_stories_read',num_stories_to_read);
                            pageSetup();
                    }).fail(function (rejection) { fail(rejection) });
                        }).fail(function (rejection) { fail(rejection) });
        }).fail(function (rejection) { fail(rejection) });
    }).fail(function(rejection){fail(rejection)});
  };

  function pageSetup(){
    $('#begin-button').addClass('disabled')
    const generation_text = ['a passage','two passages','three passages','four passages'][num_stories_to_read-1];
    $('#begin-button').removeClass('disabled')
    $('#initial_paragraph').html('In this experiment, you will read ' + generation_text + ' of text. We will then ask you some questions about what you read.')
    $('#instruct_ready_div').css('display','block');

    $('#begin-button').on('click',function(){
      $('#begin-button').addClass('disabled');
      dallinger.goToPage('exp');
    })
  }

  createParticipantAndAgent();