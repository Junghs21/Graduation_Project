import grpc
import pickle
import traceback
import grand.federated_pb2_grpc as pb2_grpc
import grand.federated_pb2 as pb2

class gRPCClient(object):
    def __init__(self, host, server_port, MAX_MESSAGE_LENGTH=2000 * 1024 * 1024):
        self.host = host
        self.server_port = server_port
        self.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

        self.channel = grpc.insecure_channel(
            '{}:{}'.format(host, server_port),
            options=[
                ('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)
            ])

        # 스텁 생성
        self.stub = pb2_grpc.grpcServiceStub(self.channel)

    def sendStates(self, states, setting, weights, drop):
        try:
            serialized_states = pickle.dumps(states)
            serialized_weights = pickle.dumps(weights)
            request = pb2.SelectedStates(state=serialized_states, setting=setting, weights=serialized_weights, drop=drop)
            return self.stub.sendState(request)
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    # 서버에 client의 학습 정보 전송
    def setup(self, data, n_class, model_name):
        request = pb2.clientInformation(data=data, n_class=n_class, model_name=model_name)
        return self.stub.valSetup(request)

    # 서버에서 global weight 수신
    def getGlobalModel(self):  
        try:
            response = self.stub.getGlobalModel(pb2.EmptyResponse(message="get global state"))
            return pickle.loads(response.state)
        except Exception as e:
            print(f"Error receiving global weight: {e}")
